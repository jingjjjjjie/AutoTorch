"""Run inference on batch CSVs and save predictions."""
import os
import re
import sys
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.idfraud.preprocessing import preprocess_csv, map_path_to_source
from data.idfraud.transforms import get_transform
from data.idfraud.dataset import IDFraudTorchDataset
from models import build_classifier_model, load_weights_from_checkpoint
from models.dino import load_dino_model


def find_checkpoints(checkpoint_folder):
    """Find all epoch_x.pt files in folder and return sorted list of (epoch_num, path)."""
    checkpoints = []
    pattern = re.compile(r'^epoch_(\d+)\.pt$')
    for fname in os.listdir(checkpoint_folder):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, os.path.join(checkpoint_folder, fname)))
    return sorted(checkpoints, key=lambda x: x[0])


def run_evaluation(cfg, checkpoint_folder, batch_list, output_path,
                   device='cuda', image_type='ori', batch_size=32, threshold=0.5):
    """
    Run inference on batches using all checkpoints in folder.

    Returns DataFrame with pred_prob_ckpt{x} columns for each checkpoint.
    """
    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_folder)
    if not checkpoints:
        raise ValueError(f"No epoch_x.pt files found in {checkpoint_folder}")
    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")

    # Preprocess data
    print("Loading data...")
    df = map_path_to_source(
        preprocess_csv(image_type=image_type, batch_list=batch_list, training_mode=False),
        training_mode=False
    )
    print(f"Samples: {len(df)}")

    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    transform = get_transform(cfg)
    dataset = IDFraudTorchDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load backbone once
    print("Loading backbone...")
    backbone = load_dino_model(cfg)

    df = df.copy()

    # Run inference for each checkpoint
    for epoch_num, ckpt_path in checkpoints:
        print(f"\nEvaluating checkpoint epoch_{epoch_num}...")
        model = build_classifier_model(cfg, device, backbone)
        model = load_weights_from_checkpoint(model, ckpt_path, device)
        model.eval()

        all_probs = []
        with torch.inference_mode():
            for X, _ in tqdm(dataloader, desc=f"Inference ckpt{epoch_num}"):
                logits = model(X.to(device)).squeeze(1)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())

        df[f'pred_prob_ckpt{epoch_num}'] = all_probs
        df[f'pred_label_ckpt{epoch_num}'] = (df[f'pred_prob_ckpt{epoch_num}'] > threshold).astype(int)

    # Save
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return df


if __name__ == "__main__":
    with open("/home/jingjie/AutoTorch/runs/Ex2_vits16_226test/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = run_evaluation(
        cfg=cfg,
        checkpoint_folder="/home/jingjie/AutoTorch/runs/Exp1_dino3vits16_v1_512/checkpoints",
        batch_list = [      
                    #"batch_datacollection_202407_production20240625none_whitebg/batch_datacollection_202407_production20240625none_whitebg.csv",
                     "2024-07-whitebg_fraud/index_annotation_mykadfront_test_set.csv",
                    #"batch_datacollection_202411_inkjet_printer/batch_datacollection_202411_inkjet_printer.csv",
                    "2024-11-inkjet_printer_eval/index_annotation_mykadfront.csv",
                    "batch_datacollection_20230214_none_recapture/batch_datacollection_20230214_recapture_eval_set.csv",
                    "batch_datacollection_20231023_wiseai_recapture_evalfixorigrotatealif/index_annotation_full.csv",
                    "batch_datacollection_20231219_wiseai_recapture_10IC/index_annotation_eval_set.csv",
                    "batch_datacollection_20240303_wiseai_recapture_whitepaperbg/index_annotation_WhitePaper_eval_set.csv",
                    "batch_datacollection_20240303_wiseai_recapture_whitepaperbg/index_annotation_WhitePaper_eval_set_square.csv",
                    "batch_datacollection_20240409_production202208redone_fullpagecolorprinteval/index_annotation_square.csv",
                    "batch_datacollection_20240409_production202208redone_fullpagecolorprinteval/index_annotation_w_genuine.csv",
                    "batch_issue_20230322_none_recapture_colorprint/index_annotation_mykadfront.csv",
                    "batch_issue_20240403_none_recapture_replaymobile/index_annotation_mykadfront.csv",
                    "batch_issue_20240408_partner_recapture_fullcolorprint/index_annotation_mykadfront.csv",
                    "batch_issue_20240412_partner_both_general/index_annotation_mykadfront.csv",
                    "batch_issue_20240704_snt_both_colorghostwhitebg/index_annotation_mykadfront.csv",
                    "batch_issue_20240704_snt_fullpagecolorprint/index_annotation_mykadfront.csv",
                    "batch_issue_20240704_snt_replay/index_annotation_mykadfront.csv",
                    "BNM_20240726/index_annotation_mykadfront_Ench12Issue.csv", 
                    "batch_issue_20240823_snt_app/index_annotation_mykadfront.csv",
                    "batch_issue_20241202_snt_mobile/index_annotation_mykadfront.csv",
                    "batch_issue_20241203_postdigi_colorprint/index_annotation_mykadfront.csv",
                    "batch_issue_set/index_annotation_mykadfront_background.csv",
                    "batch_production_202208_redone/index_annotation.csv",
                    "batch_production_20240206_none_1000/index_annotation_mykadfront_annotated_eval_set.csv",
                    "batch_production_20240405_none_merged/index_annotation.csv",
                    "batch_production_20240513_20240519_snt_random/batch_production_20240513_20240519_snt_random.csv",
                    #"batch_production_20240625_whitebg_genuine/batch_production_20240625_whitebg_genuine.csv",
                    "production_20240625_whitebg_genuine/index_annotations_mykadfront_filtered_train_set.csv",
                    "BNM_20240726/index_annotation_mykadfront_processed_filtered.csv", 
                    "batch_production_20250416_none_mypr/index_annotation_mypr_crop_eval.csv",
                    "batch_production_20250416_none_mytentera/index_annotation_mytentera_crop_eval.csv",
                    "batch_production_20250514_none_mytentera/index_annotation_mytentera_crop_eval.csv",
                    #"batch_datacollection_20250530_printed_cutout/batch_datacollection_20250530_printed_cutout.csv",
                    "2025-05-printed_cutout/index_annotation_mykadfront_eval.csv",
                    #"batch_datacollection_20250707_fakeid_v2/batch_datacollection_20250707_fakeid_v2.csv",
                    "2025-07-fakeidtester/index_annotation_mykadfront_eval_v2.csv",
                    #"batch_datacollection_202501_inkjet_printer_cutout_augmentedmaskedbw/batch_datacollection_202501_inkjet_printer_cutout_augmentedmaskedbw.csv",
                    "2025-01-inkjet_printer_cutout_eval/index_annotation_mykadfront_augmented_masked_bw.csv",         
                    "batch_production_20240206_none_1000/index_annotation_mykadfront_annotated_eval_set_augmented_masked_bw.csv",
                    #"grayscale_print_cutout_test_plan_subject/grayscale_print_cutout_test_plan_subject.csv",
                    "1.4 Grayscale Print Cutout Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"colorprint_cutout_test_plan_subject/colorprint_cutout_test_plan_subject.csv",
                    "1.5 Color Print Cutout Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"color_print_test_plan_subject/color_print_test_plan_subject.csv",
                    "1.3 Color Print Test/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"grayscale_print_test_plan_subject/grayscale_print_test_plan_subject.csv",
                    "1.2 Grayscale Print Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"color_print_with_background2_test_plan_subject/color_print_with_background2_test_plan_subject.csv",
                    "1.6.1 Color Print with Background 2 Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"replay_monitor_test_plan_subject/replay_monitor_test_plan_subject.csv",
                    "1.10 Replay Monitor Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"replay_mobile_test_plan_subject/replay_mobile_test_plan_subject.csv",
                    "1.11 Replay Mobile Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"replay_tablet_test_plan_subject/replay_tablet_test_plan_subject.csv",
                    "1.12 Replay Tablet Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    #"genuine_with_background_test_plan_subject/genuine_with_background_test_plan_subject.csv",
                    "1.1 Genuine with Background Test Plan/cleaned_filtered_annotated_index_annotation_jpg_subject_mykadfront_test.csv",
                    "20250804_myekyc_fullday/annotated_filtered_annotated_index_annotation_mykadfront_genuine.csv",                
                    "20250804_myekyc_fullday/annotated_filtered_annotated_index_annotation_mykadfront_print.csv",
                    "20250805_myekyc_fullday/annotated_index_annotation_mykadfront_cutout.csv",
                    "20250805_myekyc_fullday/annotated_index_annotation_mykadfront_genuine.csv", 
                    "20250805_myekyc_fullday/annotated_index_annotation_mykadfront_printed.csv",
                    "20250805_myekyc_fullday/annotated_index_annotation_mykadfront_replay.csv", 
                    "20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_genuine.csv",
                    "20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_print.csv",
                    "20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_replay.csv",
                    "20250808_genuine&cutout/annotated_index_annotation_genuine.csv",
                    "20250808_genuine&cutout/annotated_index_annotation_print_cutout.csv",
                    "20250808_replay_tablet/annotated_index_annotation_genuine.csv",
                    "20250808_replay_tablet/annotated_index_annotation_replay.csv",
                    "20250808_tamper_face/annotated_index_annotation_genuine.csv",
                    "20250807_genuine/annotated_index_annotation.csv",
                    "20250807_tamper_4/annotated_index_annotation.csv",
                    "20250811_genuine_ekyc/filtered_index_annotation.csv",
                    "20250811_Replay_FullPage/filtered_index_annotation_v2_print.csv",
                    "20250811_Replay_FullPage/filtered_index_annotation_v2_replay.csv",
                    "20250811_tamper_face/filtered_index_annotation.csv",
                    "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_print.csv",
                    "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_replay.csv",
                    "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_grayscale.csv",
                    "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_print.csv",
                    "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_replay.csv",
                    "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_genuine.csv",
                    "batch_issue_20250910_bmmb_both_general/index_annotation_mykadfront_v3.csv",
                    "batch_production_20250830_20250903_myid/annotated_index_annotation_mykadfront_v2_genuine.csv",
                    "batch_production_20250830_20250903_myid/annotated_index_annotation_mykadfront_v2_print.csv",
                    "batch_production_20250830_20250903_myid/annotated_index_annotation_mykadfront_v2_replay.csv",
                    ],
        output_path="predictions.csv",
        batch_size=40,
    )
