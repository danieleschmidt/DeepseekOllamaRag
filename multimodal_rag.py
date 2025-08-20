"""
Multi-Modal RAG Enhancement Module

Extends the RAG system to support multi-modal document understanding:
1. Vision-Language Integration for PDF images/charts/tables
2. Knowledge Graph Construction and Traversal
3. Multi-Document Cross-Referencing
4. Adaptive Multi-Modal Retrieval
5. Context-Aware Multi-Modal Synthesis

Research Innovation:
- Cross-modal semantic alignment for better understanding
- Dynamic knowledge graph updates from document content
- Multi-document coherence optimization
- Adaptive fusion of textual and visual information
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import networkx as nx
from PIL import Image
import fitz  # PyMuPDF for image extraction
import base64
from io import BytesIO
import json
from collections import defaultdict
import re

# For multi-modal embeddings and vision processing
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logging.warning("Vision models not available. Install transformers for full multi-modal support.")

from config import config
from utils import setup_logging
from caching import global_cache

logger = setup_logging()


@dataclass
class MultiModalDocument:
    """Enhanced document representation with multi-modal content."""
    text_content: str
    images: List[Dict[str, Any]]  # List of {image: PIL.Image, caption: str, bbox: tuple}
    tables: List[Dict[str, Any]]  # List of extracted tables
    figures: List[Dict[str, Any]]  # List of figures with captions
    metadata: Dict[str, Any]
    knowledge_entities: List[Dict[str, Any]]  # Extracted entities
    cross_references: Dict[str, List[str]]  # Cross-references to other documents


class VisionLanguageProcessor:
    """
    Vision-Language processing for multi-modal document understanding.
    """
    
    def __init__(self):
        self.vision_available = VISION_AVAILABLE
        self.image_processor = None
        self.image_captioner = None
        self.clip_processor = None
        self.clip_model = None
        
        if self.vision_available:
            self._initialize_vision_models()
    
    def _initialize_vision_models(self):
        """Initialize vision-language models."""
        try:
            # BLIP for image captioning
            self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # CLIP for image-text alignment
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            logger.info("Vision-language models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize vision models: {str(e)}")
            self.vision_available = False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF with metadata."""
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # Valid image
                            img_data = pix.tobytes("png")
                            image = Image.open(BytesIO(img_data))
                            
                            # Get image position and size
                            img_rect = page.get_image_bbox(img)
                            
                            images.append({
                                'image': image,
                                'page': page_num + 1,
                                'bbox': img_rect,
                                'size': (pix.width, pix.height),
                                'index': img_index,
                                'caption': '',  # Will be filled by captioning
                                'extracted_text': ''  # Will be filled by OCR if needed
                            })
                        
                        pix = None  # Clean up
                        
                    except Exception as e:
                        logger.warning(f"Could not extract image {img_index} from page {page_num}: {str(e)}")
            
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
        
        return images
    
    def generate_image_captions(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate captions for extracted images."""
        if not self.vision_available:
            return images
        
        enhanced_images = []
        
        for img_data in images:
            try:
                image = img_data['image']
                
                # Generate caption using BLIP
                inputs = self.image_processor(image, return_tensors="pt")
                out = self.image_captioner.generate(**inputs, max_length=50)
                caption = self.image_processor.decode(out[0], skip_special_tokens=True)
                
                img_data['caption'] = caption
                enhanced_images.append(img_data)
                
                logger.debug(f"Generated caption for image on page {img_data['page']}: {caption}")
                
            except Exception as e:
                logger.warning(f"Could not caption image: {str(e)}")
                enhanced_images.append(img_data)
        
        return enhanced_images
    
    def calculate_image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Calculate similarity between image and text using CLIP."""
        if not self.vision_available:
            return 0.0
        
        try:
            inputs = self.clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
            outputs = self.clip_model(**inputs)
            
            # Get the cosine similarity
            logits_per_image = outputs.logits_per_image
            similarity = torch.sigmoid(logits_per_image).item()
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Could not calculate image-text similarity: {str(e)}")
            return 0.0
    
    def extract_visual_features(self, image: Image.Image) -> np.ndarray:
        """Extract visual features from image using CLIP."""
        if not self.vision_available:
            return np.zeros(512)  # Return zero vector if not available
        
        try:
            inputs = self.clip_processor(images=[image], return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            return image_features.detach().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Could not extract visual features: {str(e)}")
            return np.zeros(512)


class KnowledgeGraphBuilder:
    """
    Knowledge graph construction from document content.
    """
    
    def __init__(self):
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'organization': r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Organization|University|Institute)\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s*[A-Z][A-Z]\b',
            'measurement': r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|km|g|kg|lb|°C|°F|%|MHz|GHz|MB|GB|TB)\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        self.relationship_patterns = {
            'authored_by': r'(?:written|authored|created|developed)\s+by\s+([^.]+)',
            'located_in': r'(?:located|based|situated)\s+in\s+([^.]+)',
            'part_of': r'(?:part\s+of|component\s+of|member\s+of)\s+([^.]+)',
            'related_to': r'(?:related\s+to|associated\s+with|connected\s+to)\s+([^.]+)',
            'caused_by': r'(?:caused\s+by|due\s+to|resulting\s+from)\s+([^.]+)'
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity = {
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8  # Rule-based extraction confidence
                }
                entities.append(entity)
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        for rel_type, pattern in self.relationship_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Find entities near this relationship
                match_start = match.start()
                match_end = match.end()
                
                # Look for entities within a window around the match
                window_size = 100
                nearby_entities = [
                    e for e in entities 
                    if abs(e['start'] - match_start) < window_size or abs(e['end'] - match_end) < window_size
                ]
                
                if len(nearby_entities) >= 2:
                    relationship = {
                        'type': rel_type,
                        'source_entity': nearby_entities[0],
                        'target_entity': nearby_entities[1],
                        'context': match.group(),
                        'confidence': 0.7
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def build_knowledge_graph(self, documents: List[MultiModalDocument]) -> nx.MultiDiGraph:
        """Build knowledge graph from multiple documents."""
        G = nx.MultiDiGraph()
        
        for doc_idx, doc in enumerate(documents):
            # Extract entities from text
            entities = self.extract_entities(doc.text_content)
            
            # Add entities as nodes
            for entity in entities:
                node_id = f"{entity['type']}_{hash(entity['text'])}"
                G.add_node(node_id, 
                          text=entity['text'],
                          type=entity['type'],
                          document_id=doc_idx,
                          confidence=entity['confidence'])
            
            # Extract and add relationships
            relationships = self.extract_relationships(doc.text_content, entities)
            
            for rel in relationships:
                source_id = f"{rel['source_entity']['type']}_{hash(rel['source_entity']['text'])}"
                target_id = f"{rel['target_entity']['type']}_{hash(rel['target_entity']['text'])}"
                
                G.add_edge(source_id, target_id,
                          relationship=rel['type'],
                          context=rel['context'],
                          confidence=rel['confidence'],
                          document_id=doc_idx)
        
        logger.info(f"Built knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def query_knowledge_graph(self, graph: nx.MultiDiGraph, query: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query knowledge graph for relevant information."""
        query_tokens = query.lower().split()
        relevant_results = []
        
        # Find nodes that match query terms
        matching_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_text = node_data.get('text', '').lower()
            
            # Calculate relevance score
            relevance = sum(1 for token in query_tokens if token in node_text)
            if relevance > 0:
                matching_nodes.append((node_id, relevance, node_data))
        
        # Sort by relevance
        matching_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Get subgraphs around matching nodes
        for node_id, relevance, node_data in matching_nodes[:10]:  # Top 10 matches
            # Get neighborhood
            neighbors = list(nx.ego_graph(graph, node_id, radius=max_depth).nodes())
            
            subgraph_info = {
                'central_entity': node_data,
                'related_entities': [],
                'relationships': [],
                'relevance_score': relevance
            }
            
            # Get related entities and relationships
            for neighbor_id in neighbors:
                if neighbor_id != node_id:
                    neighbor_data = graph.nodes[neighbor_id]
                    subgraph_info['related_entities'].append(neighbor_data)
                    
                    # Get edges between central node and neighbors
                    if graph.has_edge(node_id, neighbor_id):
                        edge_data = graph.get_edge_data(node_id, neighbor_id)
                        subgraph_info['relationships'].extend(list(edge_data.values()))
            
            relevant_results.append(subgraph_info)
        
        return relevant_results


class MultiDocumentProcessor:
    """
    Processor for handling multiple documents with cross-referencing.
    """
    
    def __init__(self):
        self.document_embeddings = {}
        self.cross_reference_cache = {}
    
    def process_document_collection(self, file_paths: List[str]) -> List[MultiModalDocument]:
        """Process a collection of documents with cross-referencing."""
        documents = []
        
        for i, file_path in enumerate(file_paths):
            logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_path}")
            
            try:
                doc = self._process_single_document(file_path)
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Build cross-references
        documents = self._build_cross_references(documents)
        
        return documents
    
    def _process_single_document(self, file_path: str) -> MultiModalDocument:
        """Process a single document into multi-modal representation."""
        
        # Extract text content (reuse existing logic)
        from core import DocumentProcessor
        doc_processor = DocumentProcessor()
        text_docs = doc_processor.load_document(file_path)
        text_content = "\n".join([doc.page_content for doc in text_docs])
        
        # Extract images if available
        vision_processor = VisionLanguageProcessor()
        images = vision_processor.extract_images_from_pdf(file_path)
        images = vision_processor.generate_image_captions(images)
        
        # Extract entities for knowledge graph
        kg_builder = KnowledgeGraphBuilder()
        entities = kg_builder.extract_entities(text_content)
        
        # Create multi-modal document
        multimodal_doc = MultiModalDocument(
            text_content=text_content,
            images=images,
            tables=[],  # TODO: Implement table extraction
            figures=[],  # TODO: Implement figure extraction
            metadata={
                'file_path': file_path,
                'processed_at': time.time(),
                'num_images': len(images),
                'num_entities': len(entities)
            },
            knowledge_entities=entities,
            cross_references={}
        )
        
        return multimodal_doc
    
    def _build_cross_references(self, documents: List[MultiModalDocument]) -> List[MultiModalDocument]:
        """Build cross-references between documents."""
        
        # Simple cross-referencing based on entity overlap
        for i, doc1 in enumerate(documents):
            doc1_entities = set(e['text'].lower() for e in doc1.knowledge_entities)
            
            for j, doc2 in enumerate(documents):
                if i != j:
                    doc2_entities = set(e['text'].lower() for e in doc2.knowledge_entities)
                    
                    # Find common entities
                    common_entities = doc1_entities.intersection(doc2_entities)
                    
                    if common_entities:
                        if 'related_documents' not in doc1.cross_references:
                            doc1.cross_references['related_documents'] = []
                        
                        doc1.cross_references['related_documents'].append({
                            'document_index': j,
                            'common_entities': list(common_entities),
                            'similarity_score': len(common_entities) / len(doc1_entities.union(doc2_entities))
                        })
        
        return documents
    
    def find_cross_document_context(self, query: str, documents: List[MultiModalDocument], 
                                   primary_doc_index: int, max_related: int = 3) -> List[Dict[str, Any]]:
        """Find relevant context from related documents."""
        
        if primary_doc_index >= len(documents):
            return []
        
        primary_doc = documents[primary_doc_index]
        query_tokens = set(query.lower().split())
        
        related_context = []
        
        # Get cross-references from primary document
        related_docs = primary_doc.cross_references.get('related_documents', [])
        
        # Sort by similarity score
        related_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        for related_info in related_docs[:max_related]:
            related_doc_index = related_info['document_index']
            related_doc = documents[related_doc_index]
            
            # Find relevant sections in related document
            sentences = related_doc.text_content.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_tokens = set(sentence.lower().split())
                overlap = query_tokens.intersection(sentence_tokens)
                
                if overlap:
                    relevant_sentences.append({
                        'text': sentence.strip(),
                        'relevance': len(overlap) / len(query_tokens.union(sentence_tokens))
                    })
            
            # Sort and select top sentences
            relevant_sentences.sort(key=lambda x: x['relevance'], reverse=True)
            
            if relevant_sentences:
                related_context.append({
                    'document_index': related_doc_index,
                    'document_path': related_doc.metadata.get('file_path', ''),
                    'common_entities': related_info['common_entities'],
                    'relevant_content': relevant_sentences[:3],  # Top 3 sentences
                    'similarity_score': related_info['similarity_score']
                })
        
        return related_context


class AdaptiveMultiModalRetrieval:
    """
    Adaptive retrieval that combines textual and visual information.
    """
    
    def __init__(self):
        self.vision_processor = VisionLanguageProcessor()
        self.text_weight = 0.7
        self.visual_weight = 0.3
    
    def multi_modal_search(self, query: str, multimodal_docs: List[MultiModalDocument], 
                          k: int = 5) -> List[Dict[str, Any]]:
        """Perform multi-modal search across documents."""
        
        results = []
        
        for doc_idx, doc in enumerate(multimodal_docs):
            # Text-based relevance
            text_relevance = self._calculate_text_relevance(query, doc.text_content)
            
            # Visual relevance (if images are available)
            visual_relevance = 0.0
            if doc.images:
                visual_relevance = self._calculate_visual_relevance(query, doc.images)
            
            # Combined relevance
            combined_relevance = (self.text_weight * text_relevance + 
                                self.visual_weight * visual_relevance)
            
            results.append({
                'document_index': doc_idx,
                'document': doc,
                'text_relevance': text_relevance,
                'visual_relevance': visual_relevance,
                'combined_relevance': combined_relevance,
                'relevance_breakdown': {
                    'text': text_relevance,
                    'visual': visual_relevance,
                    'knowledge_graph': 0.0  # TODO: Add KG relevance
                }
            })
        
        # Sort by combined relevance
        results.sort(key=lambda x: x['combined_relevance'], reverse=True)
        
        return results[:k]
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Calculate text-based relevance score."""
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        
        if not query_tokens or not text_tokens:
            return 0.0
        
        intersection = query_tokens.intersection(text_tokens)
        union = query_tokens.union(text_tokens)
        
        return len(intersection) / len(union)
    
    def _calculate_visual_relevance(self, query: str, images: List[Dict[str, Any]]) -> float:
        """Calculate visual relevance score."""
        if not images or not self.vision_processor.vision_available:
            return 0.0
        
        max_relevance = 0.0
        
        for img_data in images:
            # Check caption relevance
            caption = img_data.get('caption', '')
            if caption:
                caption_relevance = self._calculate_text_relevance(query, caption)
                max_relevance = max(max_relevance, caption_relevance)
            
            # Check image-text similarity using CLIP
            if 'image' in img_data:
                visual_similarity = self.vision_processor.calculate_image_text_similarity(
                    img_data['image'], query)
                max_relevance = max(max_relevance, visual_similarity)
        
        return max_relevance
    
    def adaptive_context_selection(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptively select context based on query type and content modalities."""
        
        # Analyze query to determine modality preferences
        query_analysis = self._analyze_query_modality(query)
        
        selected_context = {
            'text_chunks': [],
            'visual_elements': [],
            'knowledge_graph_info': [],
            'cross_document_references': [],
            'modality_weights': query_analysis
        }
        
        for result in search_results:
            doc = result['document']
            
            # Select text content
            if query_analysis['text_preference'] > 0.5:
                text_chunks = self._extract_relevant_text_chunks(query, doc.text_content)
                selected_context['text_chunks'].extend(text_chunks)
            
            # Select visual content
            if query_analysis['visual_preference'] > 0.3 and doc.images:
                relevant_images = self._select_relevant_images(query, doc.images)
                selected_context['visual_elements'].extend(relevant_images)
            
            # Add knowledge graph information
            if query_analysis['structured_preference'] > 0.4:
                kg_info = self._extract_kg_context(query, doc.knowledge_entities)
                selected_context['knowledge_graph_info'].extend(kg_info)
        
        return selected_context
    
    def _analyze_query_modality(self, query: str) -> Dict[str, float]:
        """Analyze query to determine modality preferences."""
        
        visual_keywords = ['image', 'picture', 'figure', 'chart', 'graph', 'diagram', 
                          'visual', 'show', 'display', 'illustration']
        structured_keywords = ['who', 'when', 'where', 'relationship', 'connected', 
                              'related', 'entity', 'organization', 'person']
        
        query_lower = query.lower()
        
        visual_score = sum(1 for keyword in visual_keywords if keyword in query_lower)
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        
        total_words = len(query.split())
        
        return {
            'text_preference': 0.7,  # Default high text preference
            'visual_preference': min(0.8, visual_score / max(total_words * 0.1, 1)),
            'structured_preference': min(0.8, structured_score / max(total_words * 0.1, 1))
        }
    
    def _extract_relevant_text_chunks(self, query: str, text: str, chunk_size: int = 200) -> List[str]:
        """Extract relevant text chunks from document."""
        words = text.split()
        chunks = []
        
        query_tokens = set(query.lower().split())
        
        for i in range(0, len(words), chunk_size // 2):  # Overlapping chunks
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunk_tokens = set(chunk_text.lower().split())
            
            # Calculate relevance
            overlap = query_tokens.intersection(chunk_tokens)
            if overlap:
                relevance = len(overlap) / len(query_tokens)
                chunks.append({
                    'text': chunk_text,
                    'relevance': relevance,
                    'start_index': i
                })
        
        # Sort by relevance and return top chunks
        chunks.sort(key=lambda x: x['relevance'], reverse=True)
        return [chunk['text'] for chunk in chunks[:3]]
    
    def _select_relevant_images(self, query: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select most relevant images based on query."""
        if not images:
            return []
        
        relevant_images = []
        
        for img_data in images:
            relevance = 0.0
            
            # Caption-based relevance
            caption = img_data.get('caption', '')
            if caption:
                relevance += self._calculate_text_relevance(query, caption)
            
            # Visual similarity
            if self.vision_processor.vision_available and 'image' in img_data:
                visual_sim = self.vision_processor.calculate_image_text_similarity(
                    img_data['image'], query)
                relevance += visual_sim
            
            if relevance > 0.2:  # Threshold for inclusion
                relevant_images.append({
                    'image_data': img_data,
                    'relevance': relevance
                })
        
        # Sort by relevance
        relevant_images.sort(key=lambda x: x['relevance'], reverse=True)
        
        return relevant_images[:2]  # Return top 2 images
    
    def _extract_kg_context(self, query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relevant knowledge graph context."""
        query_tokens = set(query.lower().split())
        relevant_entities = []
        
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            entity_tokens = set(entity_text.split())
            
            overlap = query_tokens.intersection(entity_tokens)
            if overlap:
                relevance = len(overlap) / len(query_tokens.union(entity_tokens))
                relevant_entities.append({
                    'entity': entity,
                    'relevance': relevance
                })
        
        # Sort and return top entities
        relevant_entities.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_entities[:5]


# Integration class for the existing RAG system
class MultiModalRAGEnhancer:
    """
    Main class to enhance existing RAG system with multi-modal capabilities.
    """
    
    def __init__(self, base_rag_system):
        self.base_rag = base_rag_system
        self.vision_processor = VisionLanguageProcessor()
        self.kg_builder = KnowledgeGraphBuilder()
        self.multi_doc_processor = MultiDocumentProcessor()
        self.adaptive_retrieval = AdaptiveMultiModalRetrieval()
        
        self.document_collection = []
        self.knowledge_graph = None
        
    def process_multimodal_document(self, file_path: str) -> Tuple[Any, Any, str, Dict]:
        """Process document with multi-modal enhancements."""
        
        # Get base processing results
        vector_store, qa_chain, doc_hash = self.base_rag.process_document(file_path)
        
        # Create multi-modal document
        multimodal_doc = self.multi_doc_processor._process_single_document(file_path)
        self.document_collection.append(multimodal_doc)
        
        # Update knowledge graph
        self.knowledge_graph = self.kg_builder.build_knowledge_graph(self.document_collection)
        
        # Enhanced metadata
        enhanced_metadata = {
            'multimodal_doc': multimodal_doc,
            'document_index': len(self.document_collection) - 1,
            'knowledge_graph': self.knowledge_graph,
            'multimodal_features': {
                'num_images': len(multimodal_doc.images),
                'num_entities': len(multimodal_doc.knowledge_entities),
                'vision_available': self.vision_processor.vision_available
            }
        }
        
        return vector_store, qa_chain, doc_hash, enhanced_metadata
    
    def multimodal_question_answering(self, query: str, enhanced_metadata: Dict, 
                                    baseline_qa_chain) -> Dict[str, Any]:
        """Enhanced QA with multi-modal information."""
        
        # Multi-modal search
        search_results = self.adaptive_retrieval.multi_modal_search(
            query, self.document_collection, k=5)
        
        # Adaptive context selection
        context_info = self.adaptive_retrieval.adaptive_context_selection(query, search_results)
        
        # Knowledge graph query
        kg_results = []
        if self.knowledge_graph:
            kg_results = self.kg_builder.query_knowledge_graph(self.knowledge_graph, query)
        
        # Cross-document context
        primary_doc_idx = enhanced_metadata.get('document_index', 0)
        cross_doc_context = self.multi_doc_processor.find_cross_document_context(
            query, self.document_collection, primary_doc_idx)
        
        # Generate enhanced response using baseline system with enriched context
        enriched_context = self._create_enriched_context(context_info, kg_results, cross_doc_context)
        
        # Use baseline QA with enhanced context
        baseline_response = baseline_qa_chain({"query": query})
        
        # Create enhanced response
        enhanced_response = {
            'answer': baseline_response.get("result", ""),
            'source_documents': baseline_response.get("source_documents", []),
            'multimodal_context': {
                'visual_elements': context_info['visual_elements'],
                'knowledge_graph_info': kg_results,
                'cross_document_references': cross_doc_context,
                'modality_analysis': context_info['modality_weights']
            },
            'enhancement_metrics': {
                'multimodal_retrieval_used': True,
                'knowledge_graph_used': len(kg_results) > 0,
                'cross_document_used': len(cross_doc_context) > 0,
                'visual_context_available': len(context_info['visual_elements']) > 0
            },
            'question': query,
            'timestamp': time.time()
        }
        
        return enhanced_response
    
    def _create_enriched_context(self, context_info: Dict, kg_results: List, 
                                cross_doc_context: List) -> str:
        """Create enriched context for the language model."""
        
        enriched_context = ""
        
        # Add text context
        if context_info['text_chunks']:
            enriched_context += "Text Context:\n"
            enriched_context += "\n".join(context_info['text_chunks'])
            enriched_context += "\n\n"
        
        # Add visual context descriptions
        if context_info['visual_elements']:
            enriched_context += "Visual Context:\n"
            for visual_elem in context_info['visual_elements']:
                img_data = visual_elem['image_data']
                caption = img_data.get('caption', 'No caption available')
                enriched_context += f"- Image on page {img_data.get('page', '?')}: {caption}\n"
            enriched_context += "\n"
        
        # Add knowledge graph information
        if kg_results:
            enriched_context += "Related Entities and Relationships:\n"
            for kg_result in kg_results[:3]:  # Top 3 results
                central_entity = kg_result['central_entity']
                enriched_context += f"- {central_entity.get('text', 'Unknown')} ({central_entity.get('type', 'entity')})\n"
                
                for rel in kg_result['relationships'][:2]:  # Top 2 relationships
                    enriched_context += f"  → {rel.get('relationship', 'related to')}: {rel.get('context', '')}\n"
            enriched_context += "\n"
        
        # Add cross-document information
        if cross_doc_context:
            enriched_context += "Cross-Document References:\n"
            for cross_ref in cross_doc_context[:2]:  # Top 2 references
                enriched_context += f"- Related document: {cross_ref.get('document_path', 'Unknown')}\n"
                for content in cross_ref['relevant_content'][:1]:  # Top relevant content
                    enriched_context += f"  → {content['text']}\n"
            enriched_context += "\n"
        
        return enriched_context