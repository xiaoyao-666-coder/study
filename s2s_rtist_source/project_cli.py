#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from s2s_rtist.cli import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=ROOT))
