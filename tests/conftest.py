import sys
from pathlib import Path

# Tests import `jarvis` and `run` the way the CLI does, from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
