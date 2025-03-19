# Contributing to Sauti Ya Kenya

Thank you for your interest in contributing to Sauti Ya Kenya! This document provides guidelines and instructions for contributing to the project.

## Ways to Contribute

1. **Voice Data Contributions**
   - Record Kenyan Swahili speech using our data collection tool
   - Review and validate existing recordings
   - Help create text prompts for recording sessions

2. **Code Contributions**
   - Fix bugs and issues
   - Implement new features
   - Improve documentation
   - Write tests
   - Optimize performance

3. **Language Expertise**
   - Help improve text preprocessing for Kenyan Swahili
   - Validate language detection and code-switching
   - Review and enhance pronunciation rules

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Sauti-Ya-Kenya.git
   cd Sauti-Ya-Kenya
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose
- Use descriptive variable names
- Comment complex logic

Example:
```python
def process_text(text: str) -> List[str]:
    """Process input text for TTS generation.
    
    Args:
        text: Input text to process
        
    Returns:
        List of processed tokens
        
    Raises:
        ValueError: If text is empty or invalid
    """
    if not text:
        raise ValueError("Text cannot be empty")
    # Implementation...
```

## Testing

- Write unit tests for new features
- Update existing tests when modifying code
- Run tests before submitting PR:
  ```bash
  pytest
  ```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Run all tests and ensure they pass
4. Update CHANGELOG.md with your changes
5. Submit PR with clear description of changes
6. Wait for review and address any feedback

## Voice Recording Guidelines

See the [README.md](README.md#contributing-voice-data-) for detailed voice recording guidelines.

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose.

1. Submit a pull request
2. Automated tests will run
3. Maintainers will review your code
4. Address any feedback
5. Once approved, your code will be merged

## Community Guidelines

- Be respectful and inclusive
- Help others when you can
- Ask questions if you're unsure
- Report bugs and issues
- Follow the code of conduct

## Getting Help

- Open an issue for bugs
- Discuss features in Issues
- Join our community chat
- Contact maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Contact

[Add contact information]

---

Thank you for helping make Sauti Ya Kenya better! 🎙️
