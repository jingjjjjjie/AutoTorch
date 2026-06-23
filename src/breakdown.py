"""
Compute per-recapture-subclass APCER/BPCER from an experiment directory.
Reads eval_predictions.csv, checkpoint_selection.csv, and log.csv from --run-dir.
Usage: python src/breakdown.py --run-dir /path/to/experiment
"""
import os
import argparse
import json
import pandas as pd

SUBCLASS_COL      = "Recapture_Subclass"
LABEL_COL         = "label"
THRESHOLD         = 0.5
GENUINE_SUBCLASS  = "Genuine"
PRED_CSV          = "eval_predictions.csv"
SELECTION_CSV     = "checkpoint_selection.csv"
LOG_CSV           = "log.csv"
BREAKDOWN_JSON    = "breakdown.json"


def breakdown(df, pred_col):
    df = df[df[LABEL_COL] != -1].copy()
    df = df[df[pred_col].notna()]
    df["_pred_fraud"] = df[pred_col].astype(float) > THRESHOLD

    rows = []
    for subclass, group in df.groupby(SUBCLASS_COL, dropna=False):
        if str(subclass) == GENUINE_SUBCLASS:
            scored, errors, metric = group[group[LABEL_COL] == 0], None, "BPCER"
            errors = scored["_pred_fraud"].sum()
        else:
            scored, metric = group[group[LABEL_COL] == 1], "APCER"
            errors = (~scored["_pred_fraud"]).sum()
        n = len(scored)
        rows.append({SUBCLASS_COL: subclass, "n": n, "metric": metric,
                     "errors": int(errors), "error_rate": errors / n if n else float("nan")})

    return pd.DataFrame(rows).sort_values(["metric", SUBCLASS_COL]).reset_index(drop=True)


def build_metrics(result, epoch, log_row):
    def aggregate(metric):
        sub = result[result["metric"] == metric]
        total_n, total_err = int(sub["n"].sum()), int(sub["errors"].sum())
        return round(total_err / total_n, 4) if total_n else None

    per_subclass = {
        str(row[SUBCLASS_COL]): {row["metric"]: round(float(row["error_rate"]), 4), "n": int(row["n"])}
        for _, row in result.iterrows()
    }
    return {
        "metrics": {
            "best_epoch":         epoch,
            "validation_stats":   {"loss": round(float(log_row["val_loss"]), 4),
                                   "accuracy": round(float(log_row["val_acc"]), 4)},
            "validation_results": {"Threshold": THRESHOLD,
                                   "APCER": round(float(log_row["val_apcer"]), 4),
                                   "BPCER": round(float(log_row["val_bpcer"]), 4)},
            "evaluation_results": {"Threshold": THRESHOLD,
                                   "APCER": aggregate("APCER"),
                                   "BPCER": aggregate("BPCER")},
            "per_subclass":       per_subclass,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    selection = pd.read_csv(os.path.join(args.run_dir, SELECTION_CSV))
    epoch     = int(selection.iloc[0]["epoch"])
    pred_col  = f"pred_prob_ckpt{epoch}"
    log_row   = pd.read_csv(os.path.join(args.run_dir, LOG_CSV)).iloc[epoch - 1]

    df      = pd.read_csv(os.path.join(args.run_dir, PRED_CSV))
    result  = breakdown(df, pred_col)
    output  = build_metrics(result, epoch, log_row)

    out = os.path.join(args.run_dir, BREAKDOWN_JSON)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
