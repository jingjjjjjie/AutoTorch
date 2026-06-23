import os
import pandas as pd
from utils.device import is_main_process


def select_best_checkpoint(run_dir, far_threshold=0.02):
    """
    Read log.csv from run_dir, rank epochs by quality, and write checkpoint_selection.csv.
    Primary: val_apcer < far_threshold → lowest val_bpcer. Fallback: lowest val_apcer.
    Returns { epoch, val_apcer, val_bpcer, met_ceiling } or None on non-main ranks.
    """
    if not is_main_process():
        return None

    df = pd.read_csv(os.path.join(run_dir, 'log.csv'))
    df['epoch'] = range(1, len(df) + 1)

    feasible = df[df['val_apcer'] < far_threshold]
    met_ceiling = not feasible.empty

    if met_ceiling:
        best = feasible.sort_values('val_bpcer').iloc[0]
    else:
        best = df.sort_values('val_apcer').iloc[0]
        print(f"WARNING: no checkpoint met val_apcer < {far_threshold:.2%}; "
              f"falling back to lowest val_apcer (epoch {int(best['epoch'])})")

    epoch = int(best['epoch'])
    print(f"Selected epoch {epoch}: "
          f"val_apcer={best['val_apcer']:.4f}, val_bpcer={best['val_bpcer']:.4f}")

    # Rank: feasible epochs (gate met) sorted by val_bpcer first,
    #        then infeasible epochs sorted by val_apcer — best at top, worst at bottom.
    feasible_ranked   = df[df['val_apcer'] <  far_threshold].sort_values('val_bpcer')
    infeasible_ranked = df[df['val_apcer'] >= far_threshold].sort_values('val_apcer')
    ranked = pd.concat([feasible_ranked, infeasible_ranked]).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))
    ranked['selected'] = ranked['epoch'] == epoch
    ranked.to_csv(os.path.join(run_dir, 'checkpoint_selection.csv'), index=False)

    return {
        'epoch':       epoch,
        'val_apcer':   float(best['val_apcer']),
        'val_bpcer':   float(best['val_bpcer']),
        'met_ceiling': met_ceiling,
    }
