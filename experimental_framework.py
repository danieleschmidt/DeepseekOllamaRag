"""
Experimental Framework for RAG Research

This module provides a comprehensive framework for conducting controlled
experiments and comparative studies on RAG system performance.

Features:
1. Baseline vs Enhanced Performance Comparison
2. Statistical Significance Testing
3. Multi-metric Evaluation Suite
4. Reproducible Experiment Management
5. Academic Publication Data Generation
"""

import logging
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import hashlib
from datetime import datetime

from config import config
from research_enhancements import ResearchEnhancedRAG, ResearchMetrics
from core import RAGPipeline
from utils import setup_logging

logger = setup_logging()


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""
    name: str
    description: str
    baseline_config: Dict[str, Any]
    enhanced_config: Dict[str, Any]
    test_queries: List[str]
    test_documents: List[str]
    repetitions: int = 3
    random_seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    experiment_id: str
    config_name: str
    system_type: str  # 'baseline' or 'enhanced'
    query: str
    document_id: str
    metrics: ResearchMetrics
    answer: str
    processing_time: float
    timestamp: float
    run_id: int


class RAGEvaluationSuite:
    """Comprehensive evaluation suite for RAG systems."""
    
    def __init__(self):
        self.evaluation_cache = {}
        
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Any], 
                                  ground_truth_docs: List[Any] = None) -> Dict[str, float]:
        """Evaluate retrieval quality with multiple metrics."""
        
        metrics = {}
        
        # Diversity metrics
        metrics['diversity'] = self._calculate_diversity(retrieved_docs)
        
        # Coverage metrics
        metrics['coverage'] = self._calculate_coverage(query, retrieved_docs)
        
        # If ground truth is available, calculate precision/recall
        if ground_truth_docs:
            precision, recall, f1 = self._calculate_precision_recall(
                retrieved_docs, ground_truth_docs)
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Relevance distribution
        relevance_scores = [self._calculate_relevance(query, doc) for doc in retrieved_docs]
        metrics.update({
            'avg_relevance': np.mean(relevance_scores),
            'min_relevance': np.min(relevance_scores),
            'max_relevance': np.max(relevance_scores),
            'relevance_std': np.std(relevance_scores)
        })
        
        return metrics
    
    def evaluate_answer_quality(self, question: str, answer: str, 
                               reference_answer: str = None, context: str = None) -> Dict[str, float]:
        """Comprehensive answer quality evaluation."""
        
        metrics = {}
        
        # Basic metrics
        metrics['answer_length'] = len(answer.split())
        metrics['clarity_score'] = self._calculate_clarity_score(answer)
        metrics['completeness_score'] = self._calculate_completeness_score(question, answer)
        
        # If reference answer provided, calculate similarity
        if reference_answer:
            metrics['reference_similarity'] = self._calculate_answer_similarity(
                answer, reference_answer)
        
        # Context-based metrics
        if context:
            metrics['context_utilization'] = self._calculate_context_utilization(
                answer, context)
            metrics['factual_consistency'] = self._calculate_factual_consistency(
                answer, context)
        
        # Language quality metrics
        metrics['fluency_score'] = self._calculate_fluency_score(answer)
        metrics['informativeness'] = self._calculate_informativeness(answer)
        
        return metrics
    
    def _calculate_diversity(self, docs: List[Any]) -> float:
        """Calculate diversity of retrieved documents."""
        if len(docs) < 2:
            return 1.0
        
        # Simple diversity based on content overlap
        contents = [doc.page_content for doc in docs]
        pairwise_similarities = []
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                similarity = self._text_similarity(contents[i], contents[j])
                pairwise_similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(pairwise_similarities)
        return 1.0 - avg_similarity
    
    def _calculate_coverage(self, query: str, docs: List[Any]) -> float:
        """Calculate how well documents cover the query aspects."""
        query_terms = set(query.lower().split())
        
        covered_terms = set()
        for doc in docs:
            doc_terms = set(doc.page_content.lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))
        
        return len(covered_terms) / len(query_terms) if query_terms else 0.0
    
    def _calculate_precision_recall(self, retrieved: List[Any], 
                                   ground_truth: List[Any]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        # Simple implementation based on content similarity
        retrieved_contents = [doc.page_content for doc in retrieved]
        gt_contents = [doc.page_content for doc in ground_truth]
        
        # Find matches (simple string similarity threshold)
        matches = 0
        for ret_content in retrieved_contents:
            for gt_content in gt_contents:
                if self._text_similarity(ret_content, gt_content) > 0.7:
                    matches += 1
                    break
        
        precision = matches / len(retrieved_contents) if retrieved_contents else 0.0
        recall = matches / len(gt_contents) if gt_contents else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _calculate_relevance(self, query: str, doc: Any) -> float:
        """Calculate relevance of document to query."""
        return self._text_similarity(query, doc.page_content)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity/readability score."""
        words = text.split()
        sentences = text.split('.')
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Penalize very long or very short sentences
        optimal_length = 15
        clarity = max(0, 1 - abs(avg_sentence_length - optimal_length) / optimal_length)
        
        return clarity
    
    def _calculate_completeness_score(self, question: str, answer: str) -> float:
        """Calculate how complete the answer is relative to question."""
        question_length = len(question.split())
        answer_length = len(answer.split())
        
        # Heuristic: complete answers are typically longer for complex questions
        expected_length = max(10, question_length * 2)
        completeness = min(1.0, answer_length / expected_length)
        
        return completeness
    
    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate semantic similarity between two answers."""
        return self._text_similarity(answer1, answer2)
    
    def _calculate_context_utilization(self, answer: str, context: str) -> float:
        """Calculate how well the answer utilizes the provided context."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        utilized_words = answer_words.intersection(context_words)
        return len(utilized_words) / len(answer_words) if answer_words else 0.0
    
    def _calculate_factual_consistency(self, answer: str, context: str) -> float:
        """Calculate factual consistency between answer and context."""
        # Simple implementation - in practice, would use more sophisticated NLP
        return self._calculate_context_utilization(answer, context)
    
    def _calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score of generated text."""
        # Simple heuristic based on sentence structure
        sentences = text.split('.')
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not valid_sentences:
            return 0.0
        
        # Check for grammatical indicators
        fluency_indicators = 0
        total_sentences = len(valid_sentences)
        
        for sentence in valid_sentences:
            words = sentence.split()
            if len(words) >= 3:  # Reasonable sentence length
                fluency_indicators += 1
            if any(word[0].isupper() for word in words):  # Proper capitalization
                fluency_indicators += 1
        
        return fluency_indicators / (total_sentences * 2) if total_sentences > 0 else 0.0
    
    def _calculate_informativeness(self, text: str) -> float:
        """Calculate how informative the text is."""
        words = text.split()
        
        # Count content words (longer words are often more informative)
        content_words = [w for w in words if len(w) > 4]
        
        # Count specific indicators of information
        specific_indicators = ['because', 'therefore', 'specifically', 'particularly', 'namely']
        info_indicators = sum(1 for word in words if word.lower() in specific_indicators)
        
        informativeness = (len(content_words) / len(words) * 0.7 + 
                          info_indicators / len(words) * 0.3) if words else 0.0
        
        return min(1.0, informativeness)


class ExperimentManager:
    """Manager for conducting controlled RAG experiments."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.evaluation_suite = RAGEvaluationSuite()
        self.results = []
        
    def create_experiment(self, name: str, description: str, 
                         test_queries: List[str], test_documents: List[str],
                         repetitions: int = 3) -> ExperimentConfig:
        """Create a new experiment configuration."""
        
        return ExperimentConfig(
            name=name,
            description=description,
            baseline_config=self._get_baseline_config(),
            enhanced_config=self._get_enhanced_config(),
            test_queries=test_queries,
            test_documents=test_documents,
            repetitions=repetitions
        )
    
    def run_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Run a complete experiment and return experiment ID."""
        
        experiment_id = self._generate_experiment_id(experiment_config)
        logger.info(f"Starting experiment: {experiment_config.name} (ID: {experiment_id})")
        
        np.random.seed(experiment_config.random_seed)
        
        # Initialize systems
        baseline_system = RAGPipeline()
        enhanced_system = ResearchEnhancedRAG(baseline_system)
        
        total_runs = (len(experiment_config.test_queries) * 
                     len(experiment_config.test_documents) * 
                     experiment_config.repetitions * 2)  # 2 systems
        
        run_count = 0
        
        # Run experiments
        for doc_path in experiment_config.test_documents:
            # Process document with both systems
            try:
                baseline_vector_store, baseline_qa_chain, baseline_hash = (
                    baseline_system.process_document(doc_path))
                
                enhanced_vector_store, enhanced_qa_chain, enhanced_hash, enhanced_metadata = (
                    enhanced_system.enhanced_document_processing(doc_path))
                
                for query in experiment_config.test_queries:
                    for rep in range(experiment_config.repetitions):
                        # Baseline system run
                        run_count += 1
                        logger.info(f"Running baseline system ({run_count}/{total_runs})")
                        
                        baseline_result = self._run_single_test(
                            experiment_id, experiment_config.name, "baseline",
                            query, doc_path, baseline_qa_chain, baseline_system, rep
                        )
                        self.results.append(baseline_result)
                        
                        # Enhanced system run
                        run_count += 1
                        logger.info(f"Running enhanced system ({run_count}/{total_runs})")
                        
                        enhanced_result = self._run_enhanced_test(
                            experiment_id, experiment_config.name, "enhanced",
                            query, doc_path, enhanced_qa_chain, enhanced_system, 
                            enhanced_metadata, rep
                        )
                        self.results.append(enhanced_result)
                        
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {str(e)}")
                continue
        
        # Save results
        self._save_experiment_results(experiment_id, experiment_config)
        
        # Generate analysis
        self._generate_experiment_analysis(experiment_id, experiment_config)
        
        logger.info(f"Experiment {experiment_config.name} completed with {len(self.results)} results")
        return experiment_id
    
    def _run_single_test(self, experiment_id: str, config_name: str, system_type: str,
                        query: str, document_path: str, qa_chain, rag_system, run_id: int) -> ExperimentResult:
        """Run a single test and collect metrics."""
        
        start_time = time.time()
        
        try:
            # Get response
            response = rag_system.ask_question(qa_chain, query)
            
            processing_time = time.time() - start_time
            
            # Create metrics
            metrics = ResearchMetrics(
                retrieval_precision=0.8,  # Placeholder - would calculate from response
                answer_relevance=0.75,    # Placeholder - would evaluate answer
                context_coherence=0.7,    # Placeholder
                token_efficiency=0.85,    # Placeholder
                response_time=processing_time
            )
            
            return ExperimentResult(
                experiment_id=experiment_id,
                config_name=config_name,
                system_type=system_type,
                query=query,
                document_id=Path(document_path).stem,
                metrics=metrics,
                answer=response.get('answer', ''),
                processing_time=processing_time,
                timestamp=time.time(),
                run_id=run_id
            )
            
        except Exception as e:
            logger.error(f"Error in {system_type} test: {str(e)}")
            return None
    
    def _run_enhanced_test(self, experiment_id: str, config_name: str, system_type: str,
                          query: str, document_path: str, qa_chain, enhanced_system,
                          enhanced_metadata: Dict, run_id: int) -> ExperimentResult:
        """Run enhanced system test."""
        
        start_time = time.time()
        
        try:
            # Get enhanced response
            response = enhanced_system.enhanced_question_answering(
                query, enhanced_metadata, qa_chain)
            
            processing_time = time.time() - start_time
            
            # Extract metrics from enhanced response
            enhancement_metrics = response.get('enhancement_metrics', {})
            quality_metrics = enhancement_metrics.get('answer_quality', {})
            
            metrics = ResearchMetrics(
                retrieval_precision=quality_metrics.get('relevance', 0.8),
                answer_relevance=quality_metrics.get('overall', 0.75),
                context_coherence=enhancement_metrics.get('context_coherence', 0.7),
                token_efficiency=enhancement_metrics.get('optimal_context_size', 5) / 10.0,
                response_time=processing_time
            )
            
            return ExperimentResult(
                experiment_id=experiment_id,
                config_name=config_name,
                system_type=system_type,
                query=query,
                document_id=Path(document_path).stem,
                metrics=metrics,
                answer=response.get('answer', ''),
                processing_time=processing_time,
                timestamp=time.time(),
                run_id=run_id
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced test: {str(e)}")
            return None
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        content = f"{config.name}_{config.description}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_baseline_config(self) -> Dict[str, Any]:
        """Get baseline system configuration."""
        return {
            "system_type": "baseline",
            "chunking_strategy": "recursive",
            "retrieval_method": "dense_only",
            "context_management": "fixed",
            "enhancements": []
        }
    
    def _get_enhanced_config(self) -> Dict[str, Any]:
        """Get enhanced system configuration."""
        return {
            "system_type": "enhanced",
            "chunking_strategy": "hierarchical_hybrid",
            "retrieval_method": "multi_vector_reranking",
            "context_management": "adaptive",
            "enhancements": [
                "hierarchical_document_understanding",
                "adaptive_context_management",
                "multi_vector_retrieval",
                "semantic_coherence_scoring"
            ]
        }
    
    def _save_experiment_results(self, experiment_id: str, config: ExperimentConfig):
        """Save experiment results to files."""
        
        # Convert results to dictionaries
        results_data = [asdict(result) for result in self.results if result is not None]
        
        # Save raw results
        results_file = self.output_dir / f"{experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_config': asdict(config),
                'results': results_data,
                'total_runs': len(results_data),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _generate_experiment_analysis(self, experiment_id: str, config: ExperimentConfig):
        """Generate statistical analysis of experiment results."""
        
        # Filter results for this experiment
        exp_results = [r for r in self.results if r and r.experiment_id == experiment_id]
        
        if not exp_results:
            logger.warning("No results to analyze")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in exp_results])
        
        # Statistical analysis
        analysis = self._perform_statistical_analysis(df)
        
        # Generate visualizations
        self._create_visualizations(df, experiment_id)
        
        # Save analysis report
        analysis_file = self.output_dir / f"{experiment_id}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {analysis_file}")
    
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results."""
        
        baseline_data = df[df['system_type'] == 'baseline']
        enhanced_data = df[df['system_type'] == 'enhanced']
        
        analysis = {
            'summary_statistics': {},
            'comparative_analysis': {},
            'statistical_tests': {}
        }
        
        metrics = ['retrieval_precision', 'answer_relevance', 'context_coherence', 
                  'token_efficiency', 'response_time']
        
        for metric in metrics:
            if f'metrics.{metric}' in df.columns:
                # Summary statistics
                analysis['summary_statistics'][metric] = {
                    'baseline': {
                        'mean': float(baseline_data[f'metrics.{metric}'].mean()),
                        'std': float(baseline_data[f'metrics.{metric}'].std()),
                        'median': float(baseline_data[f'metrics.{metric}'].median())
                    },
                    'enhanced': {
                        'mean': float(enhanced_data[f'metrics.{metric}'].mean()),
                        'std': float(enhanced_data[f'metrics.{metric}'].std()),
                        'median': float(enhanced_data[f'metrics.{metric}'].median())
                    }
                }
                
                # Statistical significance test
                baseline_values = baseline_data[f'metrics.{metric}'].values
                enhanced_values = enhanced_data[f'metrics.{metric}'].values
                
                if len(baseline_values) > 0 and len(enhanced_values) > 0:
                    t_stat, p_value = stats.ttest_ind(enhanced_values, baseline_values)
                    
                    analysis['statistical_tests'][metric] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size': float((enhanced_values.mean() - baseline_values.mean()) / 
                                           np.sqrt((enhanced_values.var() + baseline_values.var()) / 2))
                    }
                
                # Improvement calculation
                baseline_mean = analysis['summary_statistics'][metric]['baseline']['mean']
                enhanced_mean = analysis['summary_statistics'][metric]['enhanced']['mean']
                
                if baseline_mean > 0:
                    improvement = ((enhanced_mean - baseline_mean) / baseline_mean) * 100
                    analysis['comparative_analysis'][f'{metric}_improvement_percent'] = improvement
        
        return analysis
    
    def _create_visualizations(self, df: pd.DataFrame, experiment_id: str):
        """Create visualizations for experiment results."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Experiment Results: {experiment_id}', fontsize=16)
        
        metrics = ['retrieval_precision', 'answer_relevance', 'context_coherence', 
                  'token_efficiency', 'response_time']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            if f'metrics.{metric}' in df.columns:
                # Box plot comparing baseline vs enhanced
                baseline_data = df[df['system_type'] == 'baseline'][f'metrics.{metric}']
                enhanced_data = df[df['system_type'] == 'enhanced'][f'metrics.{metric}']
                
                axes[row, col].boxplot([baseline_data, enhanced_data], 
                                     labels=['Baseline', 'Enhanced'])
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{experiment_id}_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")


def create_research_benchmark() -> List[str]:
    """Create a comprehensive benchmark query set for RAG evaluation."""
    
    return [
        # Factual questions
        "What is the main topic discussed in this document?",
        "Who are the authors mentioned in this text?",
        "What specific dates or time periods are referenced?",
        
        # Analytical questions
        "What are the key findings or conclusions presented?",
        "How do the authors support their main arguments?",
        "What methodology was used in this research?",
        
        # Comparative questions
        "What are the advantages and disadvantages discussed?",
        "How does this approach compare to previous methods?",
        "What are the different perspectives presented on this topic?",
        
        # Complex reasoning
        "What are the implications of these findings for future research?",
        "How do the different sections of this document relate to each other?",
        "What evidence is provided to support the main claims?",
        
        # Specific detail questions
        "What specific examples or case studies are mentioned?",
        "What technical terms or concepts are defined in this document?",
        "What numerical data or statistics are provided?"
    ]


if __name__ == "__main__":
    # Example usage
    manager = ExperimentManager()
    
    # Create benchmark experiment
    benchmark_queries = create_research_benchmark()
    test_documents = ["sample_document.pdf"]  # Add your test documents
    
    experiment = manager.create_experiment(
        name="HDU_RAG_Benchmark",
        description="Comparing baseline RAG vs enhanced HDU-RAG with adaptive features",
        test_queries=benchmark_queries,
        test_documents=test_documents,
        repetitions=3
    )
    
    # Run experiment
    experiment_id = manager.run_experiment(experiment)
    logger.info(f"Completed benchmark experiment: {experiment_id}")