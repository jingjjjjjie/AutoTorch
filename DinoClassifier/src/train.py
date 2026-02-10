"""
Trains a PyTorch image classification model using device-agnostic code.
"""

# TODO  
# GPU Selection
# DDP
# Cropped, Ori model
# Merged model
# Model Selection
# Early Stopping
# Learning rate decay
# Visualize Traning plot
# Checkpoint saving
# Save CSV
# Fix utils: some file not found (FileNotFoundError)

# BUGS
# /mnt/auto-ekyc/live_data/20250811_Replay_FullPage/mykadfront/orig/20250811_123139.jpg
# Looking at fix_path(), it checks paths in this order: The function returns path even if the file doesn't exist in ANY of the locations. It should either:

import os
import torch
from torch import nn
import engine, utils
from model_utils import save_model
from model_builder import CustomClassifierModel
from data_setup import create_dataloaders
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_UNITS = 384
LEARNING_RATE = 1e-4

# model loading parameters
REPO_DIR = "/home/jingjie/dev/dino/dinov3"
CHECKPOINT_PATH = "/home/jingjie/dev/dino/DinoClassifier/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
FREEZE_BACKBONE = False
MODEL_NAME = "finetuning_test_100epoch_test.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_batch = [
                               "batch_production_202207_redone_rejectedcase/index_annotation.csv",
				# 				"batch_production_202209_redone_rejectedcase/index_annotation.csv",
				# 				"batch_legacy_20221026_subset/index_annotation_ori.csv",
				# 				"batch_datacollection_20230214_none_recapture/batch_datacollection_20230214_recapture_train_set.csv",
				# 				"batch_datacollection_20231219_wiseai_recapture_10IC/index_annotation_train_set.csv",
				# 				"batch_issue_20230322_none_recapture_colorprint/index_annotation_mykadfront.csv",
				# 				"batch_production_20240206_none_1000/index_annotation_mykadfront_annotated_train_set.csv",
				# 				"batch_datacollection_20240409_production202207redone_fullpagecolorprinttrain/index_annotation.csv",
				# 				"batch_datacollection_20240303_wiseai_recapture_whitepaperbg/index_annotation_WhitePaper_train_set_w_square.csv",
				# 				"batch_issue_20240412_partner_both_general/index_annotation_mykadfront.csv",
				# 				"batch_issue_20240704_snt_both_colorghostwhitebg/index_annotation_mykadfront.csv",
				# 				"batch_issue_20240704_snt_replay/index_annotation_mykadfront.csv",
				# 				"batch_production_20240625_whitebg_genuine/batch_production_20240625_whitebg_genuine.csv",
				# 				"batch_datacollection_202407_production20240625none_whitebg/batch_datacollection_202407_production20240625none_whitebg.csv",
				# 				"batch_weakness_20240726_bnm_colorprint/batch_weakness_20240726_bnm_colorprint.csv",
				# 				"batch_datacollection_202408_customdevicevariance/batch_datacollection_202408_customdevicevariance.csv",
				# 				"batch_datacollection_20240806_wiseai_both_whitepapercrossplatform/updated_annotations_train.csv",
				# 				"batch_issue_20240823_snt_app/index_annotation_mykadfront.csv",
				# 				"batch_datacollection_202411_inkjet_printer/batch_datacollection_202411_inkjet_printer.csv",
				# 				"batch_datacollection_202501_inkjet_printer_cutout/batch_datacollection_202501_inkjet_printer_cutout.csv",
				# 				"batch_production_20250416_none_mypr/index_annotation_mypr_crop_train.csv",
				# 				"batch_production_20250416_none_mytentera/index_annotation_mytentera_crop_train.csv",
				# 				"batch_production_20250514_none_mytentera/index_annotation_mytentera_crop_train.csv",
				# 				"batch_datacollection_20250530_printed_cutout/batch_datacollection_20250530_printed_cutout.csv",
				# 				"batch_datacollection_20250707_fakeid_v2/batch_datacollection_20250707_fakeid_v2.csv",
				# 				"batch_datacollection_202501_inkjet_printer_cutout_augmentedmaskedbw/batch_datacollection_202501_inkjet_printer_cutout_augmentedmaskedbw.csv",
				# 				"batch_production_20240206_none_1000/index_annotation_mykadfront_annotated_train_set_augmented_masked_bw.csv",
				# 				"grayscale_print_cutout_test_plan_subject/grayscale_print_cutout_test_plan_subject.csv",
				# 				"colorprint_cutout_test_plan_subject/colorprint_cutout_test_plan_subject.csv",
				# 				"color_print_test_plan_subject/color_print_test_plan_subject.csv",
				# 				"grayscale_print_test_plan_subject/grayscale_print_test_plan_subject.csv",
				# 				"color_print_with_background2_test_plan_subject/color_print_with_background2_test_plan_subject.csv",
				# 				"replay_monitor_test_plan_subject/replay_monitor_test_plan_subject.csv",
				# 				"replay_mobile_test_plan_subject/replay_mobile_test_plan_subject.csv",
				# 				"replay_tablet_test_plan_subject/replay_tablet_test_plan_subject.csv",
				# 				"genuine_with_background_test_plan_subject/genuine_with_background_test_plan_subject.csv",
				# 				"20250804_myekyc_fullday/filtered_annotated_index_annotation_mykadfront_train.csv",
				# 				"batch_issue_20250910_bmmb_both_general/index_annotation_mykadfront_v3.csv",
				# 				"20250804_myekyc_fullday/annotated_filtered_annotated_index_annotation_mykadfront_genuine.csv",
				# 				"20250804_myekyc_fullday/annotated_filtered_annotated_index_annotation_mykadfront_print.csv",
				# 				"20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_genuine.csv",
				# 				"20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_print.csv",
				# 				"20250806_myekyc_owntester/annotated_index_annotation_mykadfront_v3_replay.csv",
				# 				"20250808_genuine&cutout/annotated_index_asnnotation_genuine.csv",
				# 				"20250808_genuine&cutout/annotated_index_annotation_print_cutout.csv",
				# 				"20250808_replay_tablet/annotated_index_annotation_genuine.csv",
				# 				"20250808_replay_tablet/annotated_index_annotation_replay.csv",
				# 				"20250808_tamper_face/annotated_index_annotation_genuine.csv",
				# 				"20250807_genuine/annotated_index_annotation.csv",
				# 				"20250807_tamper_4/annotated_index_annotation.csv",
				# 				"20250811_genuine_ekyc/filtered_index_annotation.csv",
				# 				"20250811_Replay_FullPage/filtered_index_annotation_v2_print.csv",
								#"20250811_Replay_FullPage/filtered_index_annotation_v2_replay.csv",
								# "20250811_tamper_face/filtered_index_annotation.csv",
								# "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_print.csv",
								# "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_replay.csv",
								# "20250813_grayscale/annotated_annotated_index_annotation_mykadfront_grayscale.csv",
								# "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_print.csv",
								# "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_replay.csv",
								# "20250813_wiseai_myid_test-dry_run/index_annotation_mykadfront_v4_genuine.csv"
            ]

# define the transform function
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

train_dataset, valid_dataset, train_loader, valid_loader, class_names  = create_dataloaders(train_list=train_batch, transform=transform, batch_size=BATCH_SIZE)

# Load backbone
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT_PATH)

# Instantiate model with dinov3 as backbone
model = CustomClassifierModel(
    backbone_model=dinov3_vits16,
    backbone_model_output_dim=HIDDEN_UNITS,
    freeze_backbone=FREEZE_BACKBONE 
).to(device)

# Train
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_loader,
             test_dataloader=valid_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
save_model(model=model,
           target_dir="../trained_models",
           model_name=MODEL_NAME)