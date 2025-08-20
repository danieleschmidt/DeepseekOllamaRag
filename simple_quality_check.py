#!/usr/bin/env python3
"""
Simple Quality Check Script

Performs basic quality verification without external dependencies.
"""

import os
import sys
import ast
import json
import time
from pathlib import Path


def check_python_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def check_import_structure(file_path):
    """Check if imports are reasonable."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return True, imports
    except Exception as e:
        return False, str(e)


def check_file_size(file_path):
    """Check file size is reasonable."""
    size = os.path.getsize(file_path)
    size_mb = size / (1024 * 1024)
    
    if size_mb > 10:  # Files over 10MB might be problematic
        return False, f"File is very large: {size_mb:.2f}MB"
    
    return True, f"{size_mb:.2f}MB"


def count_lines_of_code(file_path):
    """Count lines of code (excluding comments and blank lines)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                comment_lines += 1
            else:
                code_lines += 1
        
        return {
            'total': total_lines,
            'code': code_lines,
            'comments': comment_lines,
            'blank': blank_lines
        }
    except Exception as e:
        return {'error': str(e)}


def check_config_validity():
    """Check if config module is valid."""
    try:
        import config
        
        # Check required attributes exist
        required_attrs = ['model', 'vector_store', 'app', 'ui']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(config.config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            return False, f"Missing config attributes: {missing_attrs}"
        
        # Check some specific values
        if config.config.vector_store.similarity_search_k <= 0:
            return False, "similarity_search_k must be positive"
        
        if config.config.app.max_file_size_mb <= 0:
            return False, "max_file_size_mb must be positive"
        
        return True, "Config validation passed"
        
    except Exception as e:
        return False, f"Config error: {str(e)}"


def run_quality_checks():
    """Run comprehensive quality checks."""
    
    print("🔍 Running Quality Checks for Research-Enhanced RAG System")
    print("=" * 70)
    
    repo_path = Path(__file__).parent
    python_files = list(repo_path.glob("*.py"))
    
    results = {
        'total_files': len(python_files),
        'syntax_valid': 0,
        'syntax_errors': [],
        'total_loc': 0,
        'file_details': {}
    }
    
    # Check each Python file
    for py_file in python_files:
        if py_file.name.startswith('.'):
            continue
            
        print(f"\n📄 Checking {py_file.name}...")
        
        file_results = {}
        
        # Syntax check
        syntax_ok, syntax_error = check_python_syntax(py_file)
        if syntax_ok:
            results['syntax_valid'] += 1
            print("  ✅ Syntax OK")
        else:
            results['syntax_errors'].append(f"{py_file.name}: {syntax_error}")
            print(f"  ❌ Syntax Error: {syntax_error}")
        
        file_results['syntax_ok'] = syntax_ok
        
        # File size check
        size_ok, size_info = check_file_size(py_file)
        print(f"  📊 Size: {size_info}")
        file_results['size_ok'] = size_ok
        file_results['size_info'] = size_info
        
        # Line count
        line_stats = count_lines_of_code(py_file)
        if 'error' not in line_stats:
            results['total_loc'] += line_stats['code']
            print(f"  📝 Lines: {line_stats['total']} total, {line_stats['code']} code")
            file_results['line_stats'] = line_stats
        
        # Import structure check (only if syntax is valid)
        if syntax_ok:
            import_ok, imports = check_import_structure(py_file)
            if import_ok:
                print(f"  📦 Imports: {len(imports)} total")
                file_results['import_count'] = len(imports)
            else:
                print(f"  ⚠️  Import analysis failed: {imports}")
        
        results['file_details'][py_file.name] = file_results
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("📊 QUALITY CHECK SUMMARY")
    print("=" * 70)
    
    print(f"📁 Total Python files: {results['total_files']}")
    print(f"✅ Syntax valid files: {results['syntax_valid']}")
    print(f"❌ Syntax error files: {len(results['syntax_errors'])}")
    print(f"📝 Total lines of code: {results['total_loc']}")
    
    syntax_percentage = (results['syntax_valid'] / results['total_files'] * 100) if results['total_files'] > 0 else 0
    print(f"🎯 Syntax success rate: {syntax_percentage:.1f}%")
    
    # Show syntax errors
    if results['syntax_errors']:
        print(f"\n❌ SYNTAX ERRORS:")
        for error in results['syntax_errors']:
            print(f"   {error}")
    
    # Config validation
    print(f"\n⚙️  CONFIG VALIDATION:")
    config_ok, config_msg = check_config_validity()
    if config_ok:
        print(f"   ✅ {config_msg}")
    else:
        print(f"   ❌ {config_msg}")
    
    # File size analysis
    print(f"\n📊 FILE SIZE ANALYSIS:")
    large_files = []
    for filename, details in results['file_details'].items():
        if not details.get('size_ok', True):
            large_files.append(filename)
    
    if large_files:
        print(f"   ⚠️  Large files: {', '.join(large_files)}")
    else:
        print(f"   ✅ All files are reasonable size")
    
    # Code complexity analysis
    print(f"\n🧮 CODE COMPLEXITY:")
    largest_files = sorted(
        [(name, details.get('line_stats', {}).get('code', 0)) 
         for name, details in results['file_details'].items()],
        key=lambda x: x[1], reverse=True
    )[:5]
    
    print(f"   📈 Largest files by LOC:")
    for filename, loc in largest_files:
        if loc > 0:
            print(f"      {filename}: {loc} lines")
    
    # Research feature verification
    print(f"\n🔬 RESEARCH FEATURE VERIFICATION:")
    research_files = [
        'research_enhancements.py',
        'multimodal_rag.py', 
        'adaptive_learning.py',
        'research_benchmarks.py',
        'experimental_framework.py',
        'research_integration.py',
        'enhanced_app.py'
    ]
    
    present_features = []
    missing_features = []
    
    for feature_file in research_files:
        if feature_file in results['file_details']:
            if results['file_details'][feature_file]['syntax_ok']:
                present_features.append(feature_file)
            else:
                missing_features.append(f"{feature_file} (syntax errors)")
        else:
            missing_features.append(f"{feature_file} (missing)")
    
    print(f"   ✅ Present features: {len(present_features)}")
    for feature in present_features:
        print(f"      - {feature}")
    
    if missing_features:
        print(f"   ❌ Missing/problematic features: {len(missing_features)}")
        for feature in missing_features:
            print(f"      - {feature}")
    
    # Overall quality assessment
    print(f"\n🏆 OVERALL QUALITY ASSESSMENT:")
    
    quality_score = 0
    max_score = 100
    
    # Syntax (40 points)
    quality_score += (syntax_percentage / 100) * 40
    
    # Research features (30 points)
    feature_percentage = (len(present_features) / len(research_files) * 100)
    quality_score += (feature_percentage / 100) * 30
    
    # Config validity (15 points)
    if config_ok:
        quality_score += 15
    
    # File sizes (10 points)
    if not large_files:
        quality_score += 10
    
    # Code volume (5 points)
    if results['total_loc'] > 1000:  # Substantial codebase
        quality_score += 5
    
    print(f"   📊 Quality Score: {quality_score:.1f}/{max_score}")
    
    if quality_score >= 85:
        print(f"   🎉 EXCELLENT - Production ready!")
    elif quality_score >= 70:
        print(f"   ✅ GOOD - Minor issues to address")
    elif quality_score >= 50:
        print(f"   ⚠️  FAIR - Several improvements needed")
    else:
        print(f"   ❌ POOR - Major issues require attention")
    
    # Success criteria
    success = (
        len(results['syntax_errors']) == 0 and
        config_ok and
        len(present_features) >= 5  # At least 5 research features working
    )
    
    print(f"\n🎯 QUALITY GATES:")
    print(f"   Syntax Errors: {'✅ PASS' if len(results['syntax_errors']) == 0 else '❌ FAIL'}")
    print(f"   Config Valid: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Research Features: {'✅ PASS' if len(present_features) >= 5 else '❌ FAIL'}")
    print(f"   Overall: {'✅ PASS' if success else '❌ FAIL'}")
    
    print("=" * 70)
    
    return success, results


if __name__ == "__main__":
    success, results = run_quality_checks()
    
    # Save results
    with open("quality_check_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to quality_check_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)