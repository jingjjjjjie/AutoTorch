#!/usr/bin/env python3
"""
Script to generate JSON configuration for fraud detection model leaderboard
by scanning checkpoint folders and automatically detecting datasets
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class DatasetConfigGenerator:
    def __init__(self):
        self.config = {
            "data": {
                "fraud_type": "idrecapture",
                "run_type": "leaderboard", 
                "model_type": "ori",
                "leaderboard": {
                    "thres_setting": "all",
                    "model_checkpoint": "",
                    "models": None,
                    "score_criteria": {
                        "weight": [],
                        "lower_as_pass": None,
                        "greater_as_pass": None
                    }
                }
            }
        }
        
        # Dataset classification rules based on naming patterns
        self.dataset_rules = {
            'production': {
                'patterns': [
                    r'batch_production_\d+.*',
                    r'\d+_myekyc_.*',
                ],
                'weights': {'apcer': -1, 'bpcer': -1},
                'note_template': 'Production dataset'
            },
            'datacollection': {
                'patterns': [
                    r'batch_datacollection_\d+.*',
                ],
                'weights': {'apcer': -1, 'bpcer': -1},
                'note_template': 'Data collection dataset'
            },
            'issue': {
                'patterns': [
                    r'batch_issue_\d+.*',
                ],
                'weights': {'apcer': -0.0769, 'bpcer': 0},
                'note_template': 'Issue test dataset'
            },
            'test_plan': {
                'patterns': [
                    r'.*test_plan_subject.*',
                    r'grayscale_print.*',
                    r'colorprint.*',
                    r'replay_.*',
                ],
                'weights': {'apcer': -1, 'bpcer': -1},
                'note_template': 'Test plan dataset'
            },
            'special_cases': {
                'batch_issue_set-index_annotation_mykadfront_background': {
                    'weights': {'apcer': 0, 'bpcer': -0.0769},
                    'note': 'Recapture test dataset'
                },
                'batch_issue_20240412_partner_both_general-index_annotation_mykadfront': {
                    'weights': {'apcer': -0.0769, 'bpcer': -0.0769},
                    'note': 'Recapture test dataset'
                }
            }
        }
    
    def scan_checkpoint_folders(self, checkpoint_path: str) -> List[str]:
        """Scan checkpoint directory for dataset folders"""
        try:
            checkpoint_dir = Path(checkpoint_path)
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
            
            # Get all directories in the checkpoint path
            folders = []
            for item in checkpoint_dir.iterdir():
                if item.is_dir():
                    folders.append(item.name)
            
            print(f"Found {len(folders)} dataset folders in {checkpoint_path}")
            return sorted(folders)
            
        except Exception as e:
            print(f"Error scanning checkpoint folders: {e}")
            return []
    
    def classify_dataset(self, folder_name: str) -> Tuple[str, Dict, str]:
        """Classify dataset based on folder name and return category, weights, and note"""
        
        # Check special cases first
        if folder_name in self.dataset_rules['special_cases']:
            special = self.dataset_rules['special_cases'][folder_name]
            return 'special', special['weights'], special['note']
        
        # Check against pattern rules
        for category, rules in self.dataset_rules.items():
            if category == 'special_cases':
                continue
                
            patterns = rules.get('patterns', [])
            for pattern in patterns:
                if re.match(pattern, folder_name, re.IGNORECASE):
                    note = self.generate_note(folder_name, rules['note_template'])
                    return category, rules['weights'], note
        
        # Default case - treat as production dataset
        return 'unknown', {'apcer': -1, 'bpcer': -1}, 'Unknown dataset type'
    
    def generate_note(self, folder_name: str, note_template: str) -> str:
        """Generate descriptive note based on folder name"""
        
        # Extract meaningful information from folder name
        if 'production' in folder_name.lower():
            if 'redone' in folder_name:
                return f"{note_template} redone"
            elif '1000' in folder_name:
                return f"{note_template} 1000 images"
            elif 'whitebg' in folder_name:
                return f"{note_template} genuine images whitebg"
            elif 'mypr' in folder_name:
                return "My PR Set"
            elif 'mytentera' in folder_name:
                return "My Tentera Set"
            elif 'colorprint' in folder_name:
                return f"{note_template} colorprint"
        
        elif 'datacollection' in folder_name.lower():
            if 'recapture' in folder_name:
                return "Recapture test dataset"
            elif 'inkjet' in folder_name:
                return "Inkjet printer dataset"
            elif 'cutout' in folder_name:
                return "Print Cutout Set"
            elif 'fakeid' in folder_name:
                return "Fake ID dataset"
            elif 'tampered' in folder_name:
                return "Tampered dataset"
        
        elif 'issue' in folder_name.lower():
            return "Recapture test dataset"
        
        elif 'test_plan' in folder_name.lower():
            if 'grayscale' in folder_name:
                return "Testing GrayScale Cutout"
            elif 'colorprint' in folder_name:
                return "Testing ColorPrint Cutout"
            elif 'replay' in folder_name:
                return "Testing Replay Attack"
        
        return note_template
    
    def add_dataset_from_folder(self, folder_name: str):
        """Add dataset configuration based on folder name"""
        category, weights, note = self.classify_dataset(folder_name)
        
        # Add APCER and BPCER entries
        self.add_dataset_weight(folder_name, "apcer", weights['apcer'], note)
        self.add_dataset_weight(folder_name, "bpcer", weights['bpcer'], note)
        
        print(f"Added dataset: {folder_name} ({category}) - {note}")
    
    def generate_from_checkpoint(self, checkpoint_path: str, model_checkpoint_name: str = None):
        """Generate complete configuration from checkpoint folder"""
        
        # # Set model checkpoint name
        # if model_checkpoint_name is None:
        #     model_checkpoint_name = self.generate_timestamp_checkpoint(
        #         os.path.basename(checkpoint_path) + "_"
        #     )
        
        self.set_leaderboard_config(model_checkpoint=model_checkpoint_name)
        
        # Scan and add all datasets
        folders = self.scan_checkpoint_folders(checkpoint_path)
        
        if not folders:
            print("No dataset folders found!")
            return
        
        print("\nProcessing datasets...")
        print("=" * 50)
        
        for folder in folders:
            self.add_dataset_from_folder(folder)
        
        print(f"\nGenerated configuration with {len(folders)} datasets")
    
    def set_basic_config(self, fraud_type: str = "idrecapture", 
                        run_type: str = "leaderboard", 
                        model_type: str = "ori"):
        """Set basic configuration parameters"""
        self.config["data"]["fraud_type"] = fraud_type
        self.config["data"]["run_type"] = run_type
        self.config["data"]["model_type"] = model_type
    
    def set_leaderboard_config(self, thres_setting: str = "all", 
                              model_checkpoint: str = "", 
                              models: Optional[List] = None):
        """Set leaderboard specific configuration"""
        self.config["data"]["leaderboard"]["thres_setting"] = thres_setting
        self.config["data"]["leaderboard"]["model_checkpoint"] = model_checkpoint
        self.config["data"]["leaderboard"]["models"] = models
    
    def add_dataset_weight(self, data_source: str, metric: str, value: float, note: str):
        """Add a dataset weight configuration"""
        weight_entry = {
            "data_source": data_source,
            "metric": metric,
            "value": value,
            "note": note
        }
        self.config["data"]["leaderboard"]["score_criteria"]["weight"].append(weight_entry)
    
    def generate_timestamp_checkpoint(self, prefix: str = "Ench19_Print_") -> str:
        """Generate a timestamp-based checkpoint name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        return f"{prefix}{timestamp}"
    
    def filter_datasets(self, include_patterns: List[str] = None, 
                       exclude_patterns: List[str] = None):
        """Filter datasets based on include/exclude patterns"""
        if include_patterns is None and exclude_patterns is None:
            return
        
        filtered_weights = []
        
        for weight in self.config["data"]["leaderboard"]["score_criteria"]["weight"]:
            data_source = weight["data_source"]
            include = True
            
            # Check include patterns
            if include_patterns:
                include = any(re.search(pattern, data_source, re.IGNORECASE) 
                            for pattern in include_patterns)
            
            # Check exclude patterns
            if exclude_patterns and include:
                include = not any(re.search(pattern, data_source, re.IGNORECASE) 
                                for pattern in exclude_patterns)
            
            if include:
                filtered_weights.append(weight)
        
        self.config["data"]["leaderboard"]["score_criteria"]["weight"] = filtered_weights
        print(f"Filtered to {len(filtered_weights)} datasets")
    
    def get_dataset_summary(self) -> Dict[str, int]:
        """Get summary of dataset types"""
        summary = {}
        weights = self.config["data"]["leaderboard"]["score_criteria"]["weight"]
        
        # Count unique datasets (since each has APCER and BPCER entries)
        unique_datasets = set()
        for weight in weights:
            if weight["metric"] == "apcer":  # Count only APCER to avoid duplicates
                unique_datasets.add(weight["data_source"])
                
                # Classify for summary
                category, _, _ = self.classify_dataset(weight["data_source"])
                summary[category] = summary.get(category, 0) + 1
        
        return summary
    
    def to_json(self, indent: int = 4) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.config, indent=indent)
    
    def save_to_file(self, filename: str, indent: int = 4):
        """Save configuration to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=indent)
        print(f"Configuration saved to {filename}")

def main():
    # Example usage
    generator = DatasetConfigGenerator()
    
    # Set basic configuration
    generator.set_basic_config()
    
    # Prompt for checkpoint path
    print("Dataset Configuration Generator")
    print("=" * 40)
    
    checkpoint_path = input("Enter checkpoint path (or press Enter for example): ").strip()
    
    if not checkpoint_path:
        # Use example with mock folder creation for demonstration
        checkpoint_path = "./example_checkpoint"
        print(f"Using example path: {checkpoint_path}")
        
        # Create example folders for demonstration
        os.makedirs(checkpoint_path, exist_ok=True)
        example_folders = [
            "batch_production_202208_redone-index_annotation_print",
            "batch_issue_20230322_none_recapture_colorprint-index_annotation_mykadfront",
            "batch_datacollection_20230214_none_recapture-batch_datacollection_20230214_recapture_eval_set",
            "grayscale_print_cutout_test_plan_subject-grayscale_print_cutout_test_plan_subject",
            "replay_mobile_test_plan_subject-replay_mobile_test_plan_subject"
        ]
        
        for folder in example_folders:
            os.makedirs(os.path.join(checkpoint_path, folder), exist_ok=True)
        
        print(f"Created example folders in {checkpoint_path}")
    
    # Generate configuration from checkpoint
    generator.generate_from_checkpoint(checkpoint_path)
    
    # Show summary
    print("\nDataset Summary:")
    print("=" * 30)
    summary = generator.get_dataset_summary()
    for category, count in summary.items():
        print(f"{category.capitalize()}: {count} datasets")
    
    # Optional filtering
    filter_choice = input("\nApply filters? (y/n): ").strip().lower()
    if filter_choice == 'y':
        include_patterns = input("Include patterns (comma-separated, or press Enter to skip): ").strip()
        exclude_patterns = input("Exclude patterns (comma-separated, or press Enter to skip): ").strip()
        
        include_list = [p.strip() for p in include_patterns.split(',')] if include_patterns else None
        exclude_list = [p.strip() for p in exclude_patterns.split(',')] if exclude_patterns else None
        
        generator.filter_datasets(include_list, exclude_list)
    
    # Save configuration
    output_file = input("\nOutput filename (default: leaderboard_config.json): ").strip()
    if not output_file:
        output_file = "leaderboard_config.json"
    
    generator.save_to_file(output_file)
    
    print(f"\nGenerated JSON preview:")
    print("=" * 50)
    print(generator.to_json())

if __name__ == "__main__":
    main()