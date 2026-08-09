"""Phases 9+11 — evaluation and validation report. Test-set MAE/R2 vs baselines, calibration by
slice, physics sanity probes (penalty rises as a parent nears / depletes; ~0 for isolated wells;
fishbone penalty rises with leg crowding), and the headline curves. Writes
reports/interference_report.md + figures. See docs/RECIPE.md Phases 9 and 11.
"""


def build_report():
    raise NotImplementedError("See docs/RECIPE.md Phases 9 and 11.")


if __name__ == "__main__":
    build_report()
