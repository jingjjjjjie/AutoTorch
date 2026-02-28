import os
import json
import pandas as pd
from glob import glob


class ModelSelection:
    """ Model Selection to sort the model by weighted scoreboard. """
    def __init__(self, score_criteria, json_file='info.json', model_type=None):
        self.model_type = model_type
        self.json_file = json_file
        self.bin_threshold = 0.5

        self.score_weight = score_criteria['weight']

        self.field_cols = []
        for entry in self.score_weight:
            entry['dataset'] = entry['data_source']
            entry['field'] = entry['data_source'] + '.' + entry['metric'] # generate field columns, for example "batch_xyz.apcer"
            if entry['field'] not in self.field_cols:
                self.field_cols.append(entry['field'])
    

    def _find_ckpt_path_by_basename(self, file_paths, target_basename):
        """Return the file path whose basename matches the target."""
        
        for full_file_path in file_paths:
            filename_with_extension = os.path.basename(full_file_path)
            filename_without_extension = os.path.splitext(filename_with_extension)[0]

            if filename_without_extension == target_basename:
                return full_file_path
            
        return None
    

    def _collect_experiment_paths(self, experiment_dir_name, runs_dir_path, checkpoint_dir_name='checkpoints', evaluation_dir_name='eval', format='*.pt'):
        """Collect evaluation directories and checkpoint file paths for an experiment.
            Returns:
            Tuple of (evaluation_epoch_dirs, checkpoint_file_paths).
        """
        experiment_dir_path = os.path.join(runs_dir_path, experiment_dir_name)

        # Collect evaluation epoch directories
        evaluation_dir_path = os.path.join(experiment_dir_path, evaluation_dir_name)
        evaluation_subdir_pattern = os.path.join(evaluation_dir_path, "*/")
        evaluation_epoch_dirs = glob(evaluation_subdir_pattern)

        # Collect checkpoint file paths
        checkpoints_dir = os.path.join(experiment_dir_path, checkpoint_dir_name)
        checkpoint_file_pattern = os.path.join(checkpoints_dir, format)
        checkpoint_file_paths = glob(checkpoint_file_pattern)

        return evaluation_epoch_dirs, checkpoint_file_paths
    

    def _extract_and_standardize_metrics_from_metrics_dict(self, raw_metrics_dict):
        """Extract and standardize raw evaluation metrics from an evaluation json.
        Args:
            raw_metrics_dict: Metric dictionary from evaluation JSON.
        """
        apcer = max(0, raw_metrics_dict["apcer"])
        bpcer = max(0, raw_metrics_dict["bpcer"])
        acer = (
            raw_metrics_dict["acer"]
            if "acer" in raw_metrics_dict
            else (apcer + bpcer) / 2
        )
        accuracy = (
            raw_metrics_dict["accuracy"]
            if "accuracy" in raw_metrics_dict
            else raw_metrics_dict["acc"]
        )
        return {
            "TP": raw_metrics_dict["TP"],
            "FP": raw_metrics_dict["FP"],
            "TN": raw_metrics_dict["TN"],
            "FN": raw_metrics_dict["FN"],
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": acer,
            "accuracy": accuracy,
        }
    

    def _build_dataset_evaluation(self, dataset_dir):
        """Build evaluation data for a single dataset directory.

        Args:
            dataset_dir (str): Path to a dataset evaluation folder.

        Returns:
            dict: Structured dict with 'metadata' and 'results' keys.
        """
        dataset_name = os.path.basename(dataset_dir)
        evaluation_json_path = os.path.join(dataset_dir, self.json_file)

        with open(evaluation_json_path) as f:
            evaluation_data = json.load(f)

        dataset_evaluation = {
            "metadata": {
                "eval_path": dataset_dir,
                "eval_csv": os.path.join(dataset_dir, f"{dataset_name}.csv"),
                "eval_info_json": evaluation_json_path,
                "threshold": self.bin_threshold
            },
            "results": {}
        }

        for threshold_name, raw_metrics_dict in evaluation_data["metrics"].items():
            dataset_evaluation["results"][threshold_name] = (
                self._extract_and_standardize_metrics_from_metrics_dict(raw_metrics_dict)
            )

        return dataset_evaluation
    

    def _build_checkpoint_records(self, eval_epoch_dirs, checkpoint_file_paths):

        checkpoint_records = []  # List to store structured results for each epoch
        # Iterate over each epoch evaluation directory
        for eval_epoch_dir in eval_epoch_dirs: # Example /Ex2_vits16_226test/eval/epoch_9/
            epoch_name = os.path.basename(os.path.normpath(eval_epoch_dir)) # Example: ".../epoch_20/" → "epoch_20"

            # Initialize the record for this epoch
            checkpoint_record = {
                "metadata": {
                    "ckpt_path": eval_epoch_dir,
                    "epoch": epoch_name,
                    "ckpt_model_path": self._find_ckpt_path_by_basename(checkpoint_file_paths, epoch_name)
                },
                "datasets": {}
            }

            dataset_pattern = os.path.join(eval_epoch_dir, "*")
            dataset_dirs = glob(dataset_pattern)

            for dataset_dir in dataset_dirs:
                dataset_name = os.path.basename(dataset_dir)
                checkpoint_record["datasets"][dataset_name] = self._build_dataset_evaluation(dataset_dir)

            checkpoint_records.append(checkpoint_record)

        return checkpoint_records


    def _calculate_weighted_score(self, row):
        """
        Calculate weighted score for a single row.

        Formula: score = Σ(m̂_i * |w_i|) / Σ(|w_i|)
        Where:   m̂_i = 1 - m_i  (lower is better, all metrics in current codebase)
        """
        score = 0
        total_weight = 0

        for weight_config in self.score_weight:
            field = weight_config['field']
            weight = weight_config['value']

            total_weight += abs(weight)
            metric_value = row[field]

            # TODO: currently only supports "lower is better" metrics
            metric_value = 1 - metric_value

            score += metric_value * abs(weight)

        return round(score / total_weight, 3)


    def _generate_leaderboard(self, checkpoint_records):
        """Generate leaderboard DataFrame from checkpoint records."""
        all_checkpoints_df = pd.DataFrame()

        for checkpoint_record in checkpoint_records: # loop through each epoch's records
            field_dataframes = []
            checkpoint_df = pd.DataFrame()

            for field in self.field_cols: # example field: "batch_xyz.apcer"
                rows = []
                dataset_name, metric = field.split('.')

                if dataset_name not in checkpoint_record['datasets']:
                    raise ValueError(f"{dataset_name} not in checkpoint {checkpoint_record['metadata']['ckpt_path']}")

                results = checkpoint_record['datasets'][dataset_name]['results']

                for threshold_key, threshold_results in results.items():
                    row = {
                        'model': checkpoint_record['metadata']['ckpt_model_path'],
                        'model_epoch': checkpoint_record['metadata']['epoch'],
                        'threshold': threshold_key,
                        field: threshold_results[metric]
                    }
                    rows.append(row)

                field_dataframes.append(pd.DataFrame(rows))

            # Merge all field dataframes for this checkpoint
            for df in field_dataframes:
                if checkpoint_df.empty:
                    checkpoint_df = df
                else:
                    checkpoint_df = pd.merge(checkpoint_df, df)

            all_checkpoints_df = pd.concat([all_checkpoints_df, checkpoint_df], axis=0)

        leaderboard_df = all_checkpoints_df
        leaderboard_df['score'] = leaderboard_df.apply(self._calculate_weighted_score, axis=1)
        leaderboard_df = leaderboard_df.sort_values(by='score', ascending=False)
        return leaderboard_df


    def run(self, eval_epoch_dirs, checkpoint_file_paths):
        """Generate leaderboard and return best model.

        Args:
            eval_epoch_dirs: Paths to evaluation epoch directories (e.g., eval/epoch_*/).
            checkpoint_file_paths: Paths to checkpoint files (e.g., checkpoints/*.pt).

        Returns:
            Tuple of (best_model, leaderboard_df).
        """
        checkpoint_records = self._build_checkpoint_records(eval_epoch_dirs, checkpoint_file_paths)
        leaderboard_df = self._generate_leaderboard(checkpoint_records)

        best_model = leaderboard_df.iloc[0, :].to_dict()

        return best_model, leaderboard_df

    @classmethod
    def run_from_config(cls, config_path, runs_dir):
        """Create ModelSelection from config file and run leaderboard generation."""
        with open(config_path) as f:
            config = json.load(f)

        leaderboard_config = config['data']['leaderboard']
        score_criteria = leaderboard_config['score_criteria']
        experiment_name = leaderboard_config['model_checkpoint']

        ms = cls(score_criteria)
        eval_dirs, ckpt_paths = ms._collect_experiment_paths(experiment_name, runs_dir)
        best, df = ms.run(eval_dirs, ckpt_paths)

        # Save leaderboard CSV
        csv_path = os.path.join(runs_dir, experiment_name, 'eval', 'leaderboard.csv')
        df.to_csv(csv_path, index=False)
        print(f"Leaderboard saved to: {csv_path}")

        return best, df


if __name__ == '__main__':
    # Resolve relative paths
    
    config_path = 'leaderboard_config.json'
    runs_dir = '/home/jingjie/DinoFT/DinoClassifier/runs'

    # Run
    best, df = ModelSelection.run_from_config(config_path, runs_dir)

    # Print results
    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by score, higher is better)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"\nBest model: {best['model_epoch']}, score: {best['score']}")

