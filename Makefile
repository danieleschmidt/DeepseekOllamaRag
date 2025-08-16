# DeepSeek RAG Application Makefile

.PHONY: help install test test-unit test-integration test-performance test-security lint format clean dev docs build deploy quality-check

# Default Python interpreter
PYTHON := python3
PIP := pip3

# Application settings
APP_NAME := deepseek-rag
VERSION := $(shell python -c "print('1.0.0')")  # Would read from version file in real app

# Help target
help:
	@echo "DeepSeek RAG Application - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install           Install all dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  install-prod      Install production dependencies only"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance benchmarks"
	@echo "  test-security     Run security tests"
	@echo "  test-coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint              Run all linting checks"
	@echo "  format            Format code with black and isort"
	@echo "  type-check        Run MyPy type checking"
	@echo "  security-scan     Run security vulnerability scans"
	@echo "  quality-check     Run all quality checks"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev               Start development server"
	@echo "  clean             Clean up generated files"
	@echo "  docs              Generate documentation"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  build             Build application package"
	@echo "  deploy-local      Deploy locally"
	@echo "  docker-build      Build Docker image"
	@echo ""

# Installation targets
install: install-prod install-dev

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install pytest pytest-cov pytest-mock pytest-benchmark
	$(PIP) install black isort flake8 mypy bandit safety
	$(PIP) install pre-commit
	@echo "Setting up pre-commit hooks..."
	pre-commit install

install-prod:
	@echo "Installing production dependencies..."
	$(PIP) install -r requirements.txt

# Testing targets
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -m "not integration and not performance and not slow" -v \
		--cov=. --cov-report=term-missing --cov-fail-under=80

test-integration:
	@echo "Running integration tests..."
	mkdir -p logs cache temp uploads
	pytest tests/ -m "integration" -v

test-performance:
	@echo "Running performance benchmarks..."
	mkdir -p logs cache temp uploads
	pytest tests/ -m "performance" -v --tb=short

test-security:
	@echo "Running security tests..."
	mkdir -p logs cache temp uploads
	pytest tests/ -m "security" -v

test-coverage:
	@echo "Running tests with detailed coverage..."
	pytest tests/ -m "not performance" \
		--cov=. --cov-report=html --cov-report=xml --cov-report=term-missing \
		--cov-fail-under=85
	@echo "Coverage report generated in htmlcov/"

# Code quality targets
lint: lint-flake8 lint-bandit

lint-flake8:
	@echo "Running Flake8 linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

lint-bandit:
	@echo "Running Bandit security linting..."
	bandit -r . -ll

format:
	@echo "Formatting code..."
	black .
	isort .

type-check:
	@echo "Running MyPy type checking..."
	mypy . --ignore-missing-imports --no-strict-optional

security-scan:
	@echo "Running security scans..."
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "Security reports generated: bandit-report.json, safety-report.json"

quality-check: format lint type-check security-scan test-unit
	@echo "All quality checks completed!"

# Development targets
dev:
	@echo "Starting development server..."
	mkdir -p logs cache temp uploads
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf junit-report.xml
	rm -rf bandit-report.json
	rm -rf safety-report.json
	rm -rf logs/*
	rm -rf cache/*
	rm -rf temp/*
	rm -rf uploads/*
	rm -rf dist/
	rm -rf build/
	@echo "Cleanup completed!"

docs:
	@echo "Generating documentation..."
	$(PYTHON) -c "
	import os
	from pathlib import Path
	
	# Create docs directory
	docs_dir = Path('docs/generated')
	docs_dir.mkdir(parents=True, exist_ok=True)
	
	# Generate module documentation
	modules = ['core', 'config', 'validation', 'security', 'caching', 'monitoring']
	
	for module in modules:
	    print(f'Generating docs for {module}...')
	    # In a real scenario, you'd use sphinx or similar
	    with open(f'docs/generated/{module}.md', 'w') as f:
	        f.write(f'# {module.title()} Module Documentation\\n\\n')
	        f.write(f'Auto-generated documentation for {module} module.\\n\\n')
	        f.write('## Classes and Functions\\n\\n')
	        f.write('TODO: Generate detailed API documentation\\n')
	
	print('Documentation generated in docs/generated/')
	"

# Build targets
build:
	@echo "Building application package..."
	mkdir -p dist/
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "Package built in dist/"

build-docker:
	@echo "Building Docker image..."
	docker build -t $(APP_NAME):$(VERSION) .
	docker tag $(APP_NAME):$(VERSION) $(APP_NAME):latest
	@echo "Docker image built: $(APP_NAME):$(VERSION)"

# Deployment targets
deploy-local: build
	@echo "Deploying locally..."
	mkdir -p logs cache temp uploads
	@echo "Local deployment ready. Run 'make dev' to start."

# Health check targets
health-check:
	@echo "Running health checks..."
	$(PYTHON) -c "
	from monitoring import health_checker
	from validation import config_validator
	
	print('Configuration validation...')
	config_result = config_validator.validate_config()
	if config_result.is_valid:
	    print('✓ Configuration is valid')
	else:
	    print('⚠ Configuration issues:', config_result.errors)
	
	print('\\nSystem health check...')
	health_results = health_checker.check_all()
	
	for component, status in health_results.items():
	    if status.status == 'healthy':
	        print(f'✓ {component}: {status.status}')
	    else:
	        print(f'⚠ {component}: {status.status} - {status.error_message or \"See details\"}')
	
	overall = health_checker.get_overall_status()
	print(f'\\nOverall system status: {overall}')
	"

# Benchmark targets
benchmark: test-performance
	@echo "Running comprehensive benchmarks..."
	$(PYTHON) -c "
	import sys
	sys.path.insert(0, 'tests')
	from test_performance import PerformanceReport
	
	print('Running all performance benchmarks...')
	report = PerformanceReport.run_all_benchmarks()
	
	print('\\n' + '='*60)
	print('PERFORMANCE BENCHMARK REPORT')
	print('='*60)
	
	for name, results in report['benchmarks'].items():
	    if 'error' in results:
	        print(f'{name}: ERROR - {results[\"error\"]}')
	    else:
	        mean_time = results.get('mean_time', 0)
	        throughput = results.get('throughput_per_second', 0)
	        print(f'{name}: {mean_time:.3f}s avg ({throughput:.1f} ops/s)')
	
	print(f'\\nTotal execution time: {report[\"summary\"][\"total_execution_time\"]:.2f}s')
	"

# CI/CD simulation
ci-test: install-dev quality-check test-integration
	@echo "CI/CD pipeline simulation completed successfully!"

# Pre-commit hooks setup
pre-commit-setup:
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	pre-commit run --all-files

# Full quality gate check (like CI)
quality-gate: clean install-dev format lint type-check security-scan test-coverage test-integration health-check
	@echo ""
	@echo "🎉 Quality gate passed! All checks completed successfully."
	@echo "Ready for deployment."

# Quick development setup
quick-setup: install-dev pre-commit-setup
	@echo "Quick development setup completed!"
	@echo "Run 'make dev' to start the development server."
	@echo "Run 'make test' to run the test suite."