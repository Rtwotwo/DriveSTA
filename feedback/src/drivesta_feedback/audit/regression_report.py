"""Summarize saved same-epoch diagnostics without changing training or labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    comparison = json.loads((root / "comparison_and_curves.json").read_text())
    v2 = json.loads((root / "v2_ep14_256/summary.json").read_text())
    v3 = json.loads((root / "v3_ep14_256/summary.json").read_text())
    failures = json.loads((root / "v3_ep14_zero418/summary.json").read_text())
    assert v2["provenance"]["tokens"] == v3["provenance"]["tokens"]
    assert failures["summary"]["scenes"] == failures["summary"]["utility"]["zeros"] == 418
    baseline_path = Path(v2["provenance"]["evaluation"])
    baseline = {
        r["token"]: r
        for r in csv.DictReader(next(baseline_path.glob("*.csv")).open())
        if r["token"] != "average"
    }
    details = []
    for row in failures["rows"]:
        token = row["token"]
        d = np.load(root / "v3_ep14_zero418" / (token + ".npz"))
        pdm, utility, y = d["pdm"], d["utility"], d["labels"][:, 6]
        qualified = d["qualified"]
        head_choice, utility_choice = pdm.argmax(), utility.argmax()
        high = y >= 0.95
        qhigh = high & qualified
        good = qualified
        acceptable = good & (y >= y[good].max() - 0.005) if good.any() else good

        # Maximum possible difference from residuals bounded in [-.05, .05].
        def gap(mask):
            if not mask.any() or mask.all():
                return None
            return float(pdm[~mask].max() - pdm[mask].max())

        g, ga = gap(high), gap(acceptable)
        details.append(
            dict(
                token=token,
                baseline_selected_pdms=100 * float(baseline[token]["score"]),
                oracle_pdms=100 * float(y.max()),
                has_high=bool(high.any()),
                has_qualified_high=bool(qhigh.any()),
                qualified_oracle_pdms=100 * float(y[qualified].max()) if qualified.any() else None,
                pdm_selected_pdms=100 * float(y[head_choice]),
                utility_selected_pdms=100 * float(y[utility_choice]),
                utility_choice_predicted_pdms=100 * float(pdm[utility_choice]),
                best_nonhigh_minus_best_high_prediction=g,
                high_unreachable_with_bounded_residual=g is not None and g >= 0.1,
                acceptable_unreachable_with_bounded_residual=ga is not None and ga >= 0.1,
                winner_margin_unattainable=ga is not None and ga > 0.08,
                head_same_index=bool(head_choice == utility_choice),
            )
        )
    summary = dict(
        scope="All 418 V3 epoch_file=14 zero-score scenes, outcome-conditioned failure analysis",
        scenes=len(details),
        has_high=sum(r["has_high"] for r in details),
        has_qualified_high=sum(r["has_qualified_high"] for r in details),
        oracle_zero=sum(r["oracle_pdms"] == 0 for r in details),
        pdm_head_also_zero=sum(r["pdm_selected_pdms"] == 0 for r in details),
        pdm_head_positive=sum(r["pdm_selected_pdms"] > 0 for r in details),
        pdm_head_high=sum(r["pdm_selected_pdms"] >= 95 for r in details),
        high_unreachable_with_bounded_residual=sum(
            r["high_unreachable_with_bounded_residual"] for r in details
        ),
        acceptable_unreachable_with_bounded_residual=sum(
            r["acceptable_unreachable_with_bounded_residual"] for r in details
        ),
        winner_margin_unattainable=sum(r["winner_margin_unattainable"] for r in details),
        selected_predicted_pdms_mean=float(
            np.mean([r["utility_choice_predicted_pdms"] for r in details])
        ),
        selected_predicted_pdms_median=float(
            np.median([r["utility_choice_predicted_pdms"] for r in details])
        ),
        v2_positive_v3_no_high=sum(
            r["baseline_selected_pdms"] > 0 and not r["has_high"] for r in details
        ),
        v2_high_v3_no_high=sum(
            r["baseline_selected_pdms"] >= 95 and not r["has_high"] for r in details
        ),
        mean_pool_oracle=float(np.mean([r["oracle_pdms"] for r in details])),
    )
    output = dict(
        full=comparison["paired_epoch14"],
        zero_decomposition=json.loads((root / "zero_score_decomposition.json").read_text()),
        random256=dict(v2=v2["summary"], v3=v3["summary"]),
        failures418=summary,
        failure_rows=details,
        limitations=[
            "One training seed per version; changed several losses together.",
            "The random256 selected-score trend differs from full NAVTEST; not a replacement estimate.",
            "Zero418 analysis conditions on V3 failures; it cannot measure total selector benefit.",
            "Safety predicates, training configuration, weights and data were not changed.",
        ],
    )
    (root / "diagnosis.json").write_text(json.dumps(output, indent=2))
    with (root / "failure418_tokens.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for version, color in [("v1", "#777777"), ("v2", "#28669b"), ("v3", "#cf652b")]:
        rows = [
            r for r in comparison["runs"][version]["evaluations"] if 11 <= r["epoch_file"] <= 14
        ]
        x = [r["epochs_completed"] for r in rows]
        axes[0].plot(x, [r["score"] for r in rows], "-o", label=version.upper(), color=color)
        axes[1].plot(x, [r["zero"] for r in rows], "-o", label=version.upper(), color=color)
    axes[0].set(ylabel="Official NAVTEST PDMS", title="Same-epoch evaluation")
    axes[1].set(ylabel="Zero-score scenes / 12,146", title="Severe failures")
    for ax in axes:
        ax.set_xlabel("Completed training epochs (file epoch + 1)")
        ax.set_xticks([12, 13, 14, 15])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(root / "regression_overview.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
