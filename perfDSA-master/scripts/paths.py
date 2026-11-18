"""Folder layout of results/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = RESULTS / "data"
SQ1 = RESULTS / "sq1"
SQ2 = RESULTS / "sq2"
PLANE = SQ2 / "plane_difference"
SQ3 = RESULTS / "sq3"
SQ4 = RESULTS / "sq4"
