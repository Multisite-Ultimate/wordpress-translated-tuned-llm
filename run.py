#!/usr/bin/env python3
"""Runner script - use this to run without pip install."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli.main import app

if __name__ == "__main__":
    app()
