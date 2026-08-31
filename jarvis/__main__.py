"""Allow `python -m jarvis ...` alongside `python run.py ...`."""

import sys

from run import main

if __name__ == "__main__":
    sys.exit(main())
