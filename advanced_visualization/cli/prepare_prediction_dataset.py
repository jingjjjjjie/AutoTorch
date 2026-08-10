"""Prepare a path-complete CSV for registered prediction and feature extraction."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ori-output", required=True, type=Path)
    parser.add_argument("--crop-output", required=True, type=Path)
    parser.add_argument("--recovered-crop-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-crop",
        action="store_true",
        help="Keep source rows whose crop cannot be recovered and omit them from crop output.",
    )
    parser.add_argument(
        "--force-label",
        type=int,
        choices=(0, 1),
        help="Assign the same binary truth label to every row.",
    )
    return parser.parse_args()


def resolve_paths(frame: pd.DataFrame, column: str, root: Path) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required image column: {column}")

    def resolve(value: object) -> str:
        path = Path(str(value)).expanduser()
        return str(path if path.is_absolute() else root / path)

    return frame[column].map(resolve)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    if "uuid" not in frame.columns:
        raise ValueError("The source CSV must contain uuid.")
    if frame["uuid"].isna().any() or frame["uuid"].astype(str).duplicated().any():
        raise ValueError("uuid must be populated and unique.")

    dataset_root = args.dataset_root.expanduser().resolve()
    frame["absolute_ori_path"] = resolve_paths(frame, "ori_path", dataset_root)
    frame["absolute_crop_path"] = resolve_paths(frame, "ocr_path", dataset_root)
    frame["absolute_ocr_path"] = frame["absolute_crop_path"]
    if args.force_label is not None:
        frame["label"] = args.force_label
    elif "label" not in frame.columns:
        if "physical_tamper_fraud" not in frame.columns:
            raise ValueError("Cannot derive label: physical_tamper_fraud is missing.")
        frame["label"] = pd.to_numeric(
            frame["physical_tamper_fraud"], errors="raise"
        ).astype(int)
    if "Recapture_Subclass" not in frame.columns:
        frame["Recapture_Subclass"] = frame.get("fraud_type", "Unknown")

    args.recovered_crop_dir.mkdir(parents=True, exist_ok=True)
    recovered = 0
    unresolved_crop: list[str] = []
    for index, row in frame.iterrows():
        crop_path = Path(row["absolute_crop_path"])
        if crop_path.is_file():
            continue
        raw_json = dataset_root / "mykadfront" / "raw" / f"{row['uuid']}.json"
        recovered_path = args.recovered_crop_dir / f"{row['uuid']}.jpg"
        try:
            payload = json.loads(raw_json.read_text(encoding="utf-8"))["card_cropped"]
            decoded = base64.b64decode(payload, validate=True)
            temporary = recovered_path.with_name(
                f".{recovered_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_bytes(decoded)
            os.replace(temporary, recovered_path)
            frame.at[index, "absolute_crop_path"] = str(recovered_path)
            frame.at[index, "absolute_ocr_path"] = str(recovered_path)
            recovered += 1
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            unresolved_crop.append(f"{row['uuid']}: {exc}")
    if unresolved_crop and not args.allow_missing_crop:
        raise FileNotFoundError(
            "Could not recover crop images:\n  " + "\n  ".join(unresolved_crop[:20])
        )
    if unresolved_crop:
        print(
            f"Warning: {len(unresolved_crop)} crop images could not be recovered; "
            "those rows will be omitted from the crop output."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    ori_frame = frame[frame["absolute_ori_path"].map(lambda value: Path(value).is_file())]
    crop_frame = frame[
        frame["absolute_crop_path"].map(lambda value: Path(value).is_file())
    ]
    args.ori_output.parent.mkdir(parents=True, exist_ok=True)
    args.crop_output.parent.mkdir(parents=True, exist_ok=True)
    ori_frame.to_csv(args.ori_output, index=False)
    crop_frame.to_csv(args.crop_output, index=False)
    print(
        f"Wrote {len(frame)} source rows to {args.output}; "
        f"{len(ori_frame)} rows have ori images and {len(crop_frame)} have crop "
        f"images ({recovered} crops recovered from raw JSON)."
    )


if __name__ == "__main__":
    main()
