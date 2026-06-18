"""Select the best checkpoint from per-epoch validation APCER/BPCER."""
import os
import pandas as pd
from utils.device import is_main_process


def select_best_checkpoint(results, run_dir, far_threshold=0.02):
    """
    Select the best epoch from per-epoch validation results.

    Hard constraint: val_apcer < far_threshold. Among epochs that satisfy it,
    pick the lowest val_bpcer. If none satisfy it, fall back to the epoch
    with the lowest val_apcer.

    Args:
        results: dict with 'val_apcer' and 'val_bpcer' lists (one entry per
            epoch, 0-indexed; checkpoint files are saved as epoch_{i+1}.pt).
        run_dir: experiment run directory, used to save the selection report.
        far_threshold: hard ceiling on val_apcer (default 2%).

    Returns:
        dict with the selected epoch (1-indexed, matching checkpoint
        filename), its val_apcer/val_bpcer, and whether the ceiling was met.
    """
    if not is_main_process():
        return None

    df = pd.DataFrame({
        'epoch': range(1, len(results['val_apcer']) + 1),
        'val_apcer': results['val_apcer'],
        'val_bpcer': results['val_bpcer'],
    })
    df['meets_ceiling'] = df['val_apcer'] < far_threshold

    passing = df[df['meets_ceiling']]
    if len(passing) > 0:
        best = passing.sort_values('val_bpcer').iloc[0]
        met_ceiling = True
    else:
        best = df.sort_values('val_apcer').iloc[0]
        met_ceiling = False
        print(f"WARNING: no checkpoint met val_apcer < {far_threshold:.2%} ceiling; "
              f"falling back to lowest val_apcer (epoch {int(best['epoch'])}, "
              f"val_apcer={best['val_apcer']:.4f})")

    print(f"Selected epoch {int(best['epoch'])}: "
          f"val_apcer={best['val_apcer']:.4f}, val_bpcer={best['val_bpcer']:.4f} "
          f"(ceiling met: {met_ceiling})")

    df['selected'] = df['epoch'] == int(best['epoch'])
    output_path = os.path.join(run_dir, 'checkpoint_selection.csv')
    df.to_csv(output_path, index=False)
    print(f"Selection report saved to {output_path}")

    return {
        'epoch': int(best['epoch']),
        'val_apcer': float(best['val_apcer']),
        'val_bpcer': float(best['val_bpcer']),
        'met_ceiling': met_ceiling,
    }
