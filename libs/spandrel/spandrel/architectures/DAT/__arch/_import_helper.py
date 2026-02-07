"""Import helper for running scripts directly from __arch directory."""
import sys
import os

# Add current directory to path for direct imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)