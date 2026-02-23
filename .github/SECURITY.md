# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**DO NOT** open public issues for security vulnerabilities.

Email: santiago@topologica.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

Expected response time: 48 hours.

## Security Best Practices for Users

- **Never** commit API keys to version control
- Use environment variables for `SEMANTIC_SCHOLAR_API_KEY`
- Keep this package updated to the latest version
- Review [CHANGELOG.md](../CHANGELOG.md) for security-related patches

## Security Features

- No API keys stored in code — environment variables only
- Input validation on all tool parameters
- Semantic Scholar API rate limits respected automatically
- HTTPS-only API communication
- Minimal dependency footprint to reduce attack surface
