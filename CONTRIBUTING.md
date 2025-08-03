# Contributing to DeepseekOllamaRag

We welcome contributions to the DeepseekOllamaRag project! This document provides guidelines for contributing to our privacy-focused RAG system.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

1. Check existing issues before creating a new one
2. Use clear, descriptive titles
3. Include system information (OS, Python version, etc.)
4. Provide steps to reproduce the issue
5. Include error messages and logs

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following our coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Commit with clear, descriptive messages
7. Push to your fork and submit a pull request

### Pull Request Guidelines

- **Title**: Use clear, descriptive titles
- **Description**: Explain what changes were made and why
- **Testing**: Include test results and coverage information
- **Documentation**: Update relevant documentation
- **Breaking Changes**: Clearly mark any breaking changes

## Development Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running
- DeepSeek R1 model available

### Local Development

```bash
# Clone your fork
git clone https://github.com/your-username/DeepseekOllamaRag.git
cd DeepseekOllamaRag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run the application
streamlit run app.py
```

## Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Quality

- **Linting**: Code must pass `flake8` and `pylint` checks
- **Formatting**: Use `black` for code formatting
- **Type Checking**: Use `mypy` for static type checking
- **Testing**: Maintain >90% test coverage

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstring format
- Update README.md for user-facing changes
- Add ADRs for significant architectural decisions

## Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_document_processing.py
│   ├── test_embeddings.py
│   └── test_retrieval.py
├── integration/
│   ├── test_api_endpoints.py
│   └── test_workflow.py
└── fixtures/
    ├── sample_documents/
    └── test_data.py
```

### Test Requirements

- Write unit tests for all business logic
- Include integration tests for workflows
- Use meaningful test names
- Mock external dependencies
- Test both success and failure scenarios

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_document_processing.py

# Run tests matching pattern
pytest -k "test_embedding"
```

## Security Guidelines

### Data Privacy

- All processing must remain local
- No external API calls without explicit user consent
- Temporary files must be cleaned up after processing
- Sensitive data should never be logged

### Code Security

- Validate all user inputs
- Use secure file handling practices
- Follow OWASP security guidelines
- Report security vulnerabilities privately

## Performance Guidelines

### Optimization Principles

- Profile before optimizing
- Prefer readable code over premature optimization
- Use appropriate data structures
- Implement proper error handling

### Memory Management

- Clean up temporary files and variables
- Use generators for large datasets
- Monitor memory usage during processing
- Implement proper resource cleanup

## Documentation Standards

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Document configuration options

### User Documentation

- Write clear, step-by-step instructions
- Include screenshots for UI changes
- Provide troubleshooting guides
- Maintain FAQ sections

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git

## Community Guidelines

### Communication

- Be respectful and inclusive
- Use clear, professional language
- Provide constructive feedback
- Help newcomers get started

### Issue Triage

- Label issues appropriately
- Provide helpful responses
- Close duplicate issues
- Update issue status regularly

## Getting Help

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community chat
- **Documentation**: Check existing docs first
- **Code Review**: Request reviews for complex changes

### Maintainer Response Times

- **Critical Bugs**: 24-48 hours
- **General Issues**: 3-5 business days
- **Feature Requests**: 1-2 weeks
- **Pull Reviews**: 2-3 business days

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

Thank you for contributing to DeepseekOllamaRag! Your contributions help make document intelligence more accessible and privacy-focused.