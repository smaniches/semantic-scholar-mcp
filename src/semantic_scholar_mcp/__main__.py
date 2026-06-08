"""
Entry point for running as module: python -m semantic_scholar_mcp

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
Organization: TOPOLOGICA LLC (https://topologica.ai)
"""

from .server import main

# Run-as-script guard; main() is tested directly, so the guard is excluded.
if __name__ == "__main__":  # pragma: no cover
    main()
