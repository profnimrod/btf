"""Sign a bundle directory with the offline registry key (Ch. 19)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle.bundle import seal
if __name__ == "__main__":
    d = sys.argv[1]
    seal(d, Path(d).name)
    print(f"[sign] {d} sealed and signed")
