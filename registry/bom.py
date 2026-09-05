"""Alias: `registry/registry.py bom` (see App. F)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from registry.registry import main
if __name__ == "__main__":
    sys.argv.insert(1, "bom")
    main()
