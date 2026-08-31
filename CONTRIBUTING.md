# Contributing to RL-Deploy-Bench

Thank you for your interest in contributing to RL-Deploy-Bench! This document
outlines the guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/rl-deploy-bench.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to the branch: `git push origin feature/your-feature`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,sb3,nvidia]"

# Run tests
pytest tests/ -v

# Run linter
black src/ tests/
isort src/ tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Add type hints to all public functions
- Write docstrings for all public modules, classes, and functions

## Pull Request Guidelines

1. **Title**: Use clear, descriptive titles (e.g., `feat: add TensorRT FP8 support`)
2. **Description**: Explain what the PR does and why
3. **Tests**: Add tests for new functionality
4. **Documentation**: Update README and docstrings as needed
5. **Keep it focused**: One PR should address one feature or fix

## Commit Message Format

We use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style (formatting, missing semicolons, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Build process, tooling, or auxiliary tools

## Reporting Issues

When reporting issues, please include:

1. Python version
2. Operating system
3. GPU model (if applicable)
4. Steps to reproduce
5. Expected behavior
6. Actual behavior
7. Error messages (if any)

## Feature Requests

We welcome feature requests! Please open an issue with:

1. Clear description of the feature
2. Why it would be useful
3. Potential implementation approach (if you have one)

## Questions?

Feel free to open an issue or reach out if you have any questions.

Thank you for contributing!
