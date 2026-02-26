# Contributing to ECGEN

Thank you for your interest in contributing to ECGEN! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-capable GPU (recommended)
- Familiarity with PyTorch and PyTorch Lightning

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/yourusername/ECGEN.git
cd ECGEN
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**:
```bash
pip install -e ".[dev,docs]"
```

4. **Verify installation**:
```bash
python scripts/test_models_only.py
```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or updates

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add type hints to all functions
- Write docstrings for public APIs
- Add tests for new functionality

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Check code style
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add support for 12-lead ECGs"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions or updates
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Screenshots (if applicable)
- Test results

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use `isort` for organizing
- **Formatting**: Use `black` for automatic formatting
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

Example:

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Brief description of the model.
    
    Longer description if needed.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # ... implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # ... implementation
        return output
```

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples for complex functions
- Keep docstrings up to date with code changes

### Testing Guidelines

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example:

```python
import pytest
import torch

from ecgen.models.pulse2pulse import WaveGANGenerator


def test_generator_forward_pass():
    """Test generator forward pass with valid input."""
    generator = WaveGANGenerator(
        model_size=50,
        num_channels=8,
        seq_length=5000,
    )
    
    # Create random input
    batch_size = 4
    noise = torch.randn(batch_size, 8, 5000)
    
    # Forward pass
    output = generator(noise)
    
    # Check output shape
    assert output.shape == (batch_size, 8, 5000)
    
    # Check output range (tanh activation)
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_generator_invalid_input():
    """Test generator raises error with invalid input."""
    generator = WaveGANGenerator(
        model_size=50,
        num_channels=8,
        seq_length=5000,
    )
    
    # Invalid input shape
    noise = torch.randn(4, 12, 5000)  # Wrong number of channels
    
    with pytest.raises(RuntimeError):
        generator(noise)
```

## Documentation

### Updating Documentation

When adding new features or changing existing ones:

1. **Update docstrings** in the code
2. **Update relevant .md files** in `docs/`
3. **Add examples** if applicable
4. **Update changelog** in `docs/development/changelog.md`

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Link to related documentation
- Keep it up to date

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** as needed
3. **Add entry to changelog** for significant changes
4. **Request review** from maintainers
5. **Address review comments**
6. **Squash commits** if requested
7. **Wait for approval** before merging

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal example to reproduce
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, GPU, etc.
- **Error messages**: Full error traceback

Example:

```markdown
**Bug Description**
Generator produces NaN values after 50 epochs.

**Steps to Reproduce**
1. Train model with default config
2. Wait until epoch 50
3. Check generated samples

**Expected Behavior**
Generated samples should be valid ECG signals.

**Actual Behavior**
All values are NaN.

**Environment**
- Python: 3.10
- PyTorch: 2.0.0
- CUDA: 11.8
- GPU: RTX 3090

**Error Message**
```
RuntimeError: NaN detected in generator output
```
```

### Feature Requests

When requesting features, include:

- **Description**: Clear description of the feature
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered

## Code Review Process

### For Contributors

- Be responsive to feedback
- Ask questions if unclear
- Make requested changes promptly
- Be patient and respectful

### For Reviewers

- Be constructive and respectful
- Explain reasoning for requested changes
- Approve when ready
- Provide clear feedback

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Give credit where due
- Follow the code of conduct

## Questions?

- **Documentation**: Check the [docs](https://yourusername.github.io/ECGEN)
- **Issues**: Search existing issues first
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers directly for sensitive issues

## License

By contributing to ECGEN, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ECGEN! 🎉
