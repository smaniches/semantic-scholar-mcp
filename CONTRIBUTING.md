# Contributing to Semantic Scholar MCP

Thank you for your interest in contributing! This project is maintained by [TOPOLOGICA LLC](https://topologica.ai).

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/semantic-scholar-mcp.git
   cd semantic-scholar-mcp
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development

### Running Tests
```bash
pytest
```

### Type Checking
```bash
mypy src/
```

### Linting
```bash
ruff check src/
```

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Add yourself to CONTRIBUTORS.md (optional)
4. Submit PR with clear description

## Code Style

- Use type hints
- Follow PEP 8
- Write docstrings for public functions
- Keep functions focused and small

## Reporting Issues

- Check existing issues first
- Provide reproduction steps
- Include error messages and logs

## Contact

- **Author:** Santiago Maniches
- **Email:** santiago@topologica.ai
- **Website:** https://topologica.ai

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
