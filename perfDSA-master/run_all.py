"""Run the pipeline in order; outputs go to results/ (layout in scripts/paths.py).

    python run_all.py                    # from postprocess on; long steps re-use saved results
    python run_all.py --from raw_model   # also rerun perfDSA on the DICOMs (GPU, hours)
    python run_all.py --from qc          # after changing a QC threshold
    python run_all.py --full             # recompute the truncation simulation and AIF swap
    python run_all.py --only sq2 sq2_slides
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "data"
DATA = [str(ROOT.parent / "PRAISE"), str(ROOT.parent / "PRAISE_BATCH_2")]

# name -> (command, extra args when the long computation is skipped)
STEPS = {
    # data -> perfusion maps
    "raw_model": (["run_dataset.py", "-i", *DATA, "-o", str(OUT / "raw_model"), "--save-maps", "--ica-full", "model"], []),
    "raw_grown": (["run_dataset.py", "-i", *DATA, "-o", str(OUT / "raw_grown"), "--save-maps", "--ica-full", "grown"], []),
    # filtered data
    "postprocess": (["scripts/postprocess_combined.py", str(OUT)], []),
    "select": (["scripts/select_runs.py", str(OUT)], []),
    "build_features": (["scripts/build_features.py", str(OUT)], []),
    "qc": (["scripts/aif_qc.py", str(OUT)], []),
    "taxonomy": (["scripts/aif_taxonomy.py", str(OUT)], []),
    "truncation": (["scripts/aif_truncation.py", str(OUT)], ["--report-only"]),
    "summary": (["scripts/data_summary.py", str(OUT)], []),
    "plane": (["scripts/plane_difference.py", str(OUT)], []),
    "features": (["scripts/aif_feature_set.py", str(OUT)], []),
    # sub-questions
    "sq1": (["scripts/sq1_full_vs_top.py", str(OUT)], []),
    "sq2": (["scripts/sq2_plane.py", str(OUT)], []),
    "sq3": (["scripts/sq3_variance.py", str(OUT)], []),
    "sq3_all_variants": (["scripts/sq3_variance.py", str(OUT), "--variants", "ica_top", "ica_full_model", "ica_full_grown"], []),
    "sq3_component_ci": (["scripts/sq3_component_ci.py", str(OUT)], []),
    "sq4": (["scripts/sq4_aif_swap.py", str(OUT)], ["--report-only"]),
    # figures
    "sq1_figures": (["scripts/sq1_summary_figure.py", str(OUT)], []),
    "sq1_slides": (["scripts/sq1_slides.py", str(OUT)], []),
    "sq2_slides": (["scripts/sq2_slides.py", str(OUT)], []),
    "sq3_figure": (["scripts/sq3_report_figure.py"], []),
    "sq3_slides": (["scripts/sq3_slide_figures.py", str(OUT)], []),
    "sq4_slides": (["scripts/sq4_slides.py", str(OUT)], []),
}
DEFAULT_START = "postprocess"


def run(names, full):
    for name in names:
        cmd, quick = STEPS[name]
        args = [sys.executable, *cmd] + ([] if full else quick)
        print(f"\n=== {name}: {' '.join(args[1:])}", flush=True)
        subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", choices=list(STEPS), default=DEFAULT_START)
    p.add_argument("--only", nargs="+", choices=list(STEPS))
    p.add_argument("--full", action="store_true", help="Recompute the long simulations (truncation, AIF swap).")
    args = p.parse_args()
    run(args.only or list(STEPS)[list(STEPS).index(args.start):], args.full)


if __name__ == "__main__":
    main()
