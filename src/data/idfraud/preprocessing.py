'''
Description: Data utilities for loading and preprocessing training batches
** currently resolves the absolute paths of image to MNT4PATH,
** currently resolves and checks the csv batches in MNT3PATH
'''
import os
import pandas as pd
from tqdm import tqdm
from utils.device import is_main_process
from sklearn.model_selection import train_test_split

MNT3PATH = '/mnt3/auto-ekyc/idrecapture/datasets'
MNT4PATH = '/mnt4/auto-ekyc/idrecapture/datasets'

def map_path_to_source(df, source_path=MNT4PATH, training_mode=True):
    """Resolves image paths in the dataframe to their absolute locations.

    Args:
        - df: DataFrame containing a 'path' column with relative image paths.
        - source_path: Base directory used to resolve the image paths.
        - training_mode: If True, assumes distributed training (DDP)
          and shows a tqdm progress bar only on the main process. Defaults to True

    Returns:
        pd.DataFrame: DataFrame with resolved absolute paths in the 'path' column.

    Raises:
        FileNotFoundError: If any image paths could not be resolved in the source directory.
    """
    missing_paths = []

    # map the relative path from the path column to the source location
    def map(x):
        path = os.path.join(source_path, x)
        if os.path.exists(path):
            return path
        missing_paths.append(x)
        return None

    # if training mode is set, check if is main process of DDP, show progress if yes. Show tqdm progress in eval
    show_progress = not training_mode or is_main_process()

    tqdm.pandas(desc="Mapping paths")
    df['path'] = df['path'].progress_apply(map) if show_progress else df['path'].apply(map)

    if missing_paths:
        raise FileNotFoundError(f"Missing paths: {missing_paths}")

    return df


def preprocess_csv(image_type, batch_list, training_mode=True, sample_fraction=1.0):
    """Reads and combines batch CSVs into a single DataFrame.

    Args:
        image_type: Image column to use for path resolution. One of 'ori' or 'crop'.
        batch_list: List of relative batch CSV paths from the config.
        training_mode: If True, assumes DDP and shows tqdm progress only on rank 0. Defaults to True.
        sample_fraction: Fraction of data to keep (0.0-1.0). Stratified by label. Defaults to 1.0 (all data).

    Returns:
        pd.DataFrame: Combined DataFrame with 'path', 'label', 'batch_directory', and 'filename' columns.

    Raises:
        Exception: If batch_list is empty or contains duplicate entries.
        ValueError: If image_type is not one of 'ori', 'crop', or 'corner'.
        FileNotFoundError: If any batch CSV paths could not be found in the source directory.
    """

    def _check_duplicate_or_empty(batch_list):
        """Validates that the batch list (from the config) is non-empty and contains no duplicates.

        Args:
            batch_list: List of batch CSV paths from the config.

        Raises:
            Exception: If batch_list is empty or contains duplicate entries.
        """
        if not batch_list:
            raise Exception("Error in preprocess_csv - Batch list is empty.")
        if len(batch_list) != len(set(batch_list)):
            raise Exception("Error in preprocess_csv - Duplicate batches inside config file.")

    def _combine_batch(image_type, batch_list):
        """Reads each batch CSV and concatenates them into one DataFrame."""
        missing_batches = []
        batch_datas = []
        for batch in tqdm(batch_list, desc="Processing batches", disable=not show_progress):

            # Example batch: "batch_issue_20240704_snt_both_colorghostwhitebg/index_annotation_mykadfront.csv"
            batch_dir = os.path.dirname(batch)  # returns batch_issue_20240704_snt_both_colorghostwhitebg/

            # join the batch_csv_path with the source directory
            batch_csv_path = os.path.join(MNT3PATH, batch)
            if os.path.exists(batch_csv_path):  # check if the batch_csv exists in the source directory
                batch_data = pd.read_csv(batch_csv_path)

                # resolves paths, create a path column, and map image type to the correct path column
                if image_type == 'crop':
                    batch_data['path'] = batch_data.apply(lambda x: os.path.join(batch_dir, x['ocr_path']), axis=1)
                elif image_type == 'ori':
                    batch_data['path'] = batch_data.apply(lambda x: os.path.join(batch_dir, x['ori_path']), axis=1)
                else:
                    raise ValueError(f"Unsupported image_type '{image_type}'. Supported types: 'crop', 'ori'.")

                # Sample this batch if fraction < 1.0 (keep at least 1)
                if sample_fraction < 1.0:
                    n_samples = max(1, round(len(batch_data) * sample_fraction))
                    batch_data = batch_data.sample(n=n_samples, random_state=42)

                batch_data['original_batch_name'] = batch  # original batch path from config
                batch_data['batch_directory'] = batch_dir  # the name of the batch (directory name)
                batch_data['filename'] = batch_data['path'].apply(os.path.basename)  # the image file name
                batch_datas.append(batch_data)
            else:  # if the csv is not found in the source directory
                missing_batches.append(batch)

        if missing_batches:
            missing_file = 'missing_batches.txt'
            with open(missing_file, 'w') as f:
                f.write('\n'.join(missing_batches))
            raise FileNotFoundError(f"Missing {len(missing_batches)} batches. Saved to {missing_file}")
        main_data = pd.concat(batch_datas, ignore_index=True)

        # map the labels: genuine = class 0, fraud = class 1
        main_data['label'] = main_data['fraud_type'].apply(lambda x: 0 if x == 'genuine' else 1)

        return main_data, missing_batches

    # if training mode is set, check if is main process of DDP, show progress if yes. Show tqdm progress in eval
    show_progress = not training_mode or is_main_process()
    _check_duplicate_or_empty(batch_list)
    main_data, missing_batches = _combine_batch(image_type, batch_list)

    return main_data


def split_data(main_data, train_val_split=0.9, csv_label_column='label', random_state=42):
    """Split a dataset into stratified train and validation sets.

    Performs a stratified split based on the specified label column so that
    class distribution is preserved in both subsets. Adds a new column
    ``dataset_type`` indicating whether each row belongs to the train or
    validation split, then returns the combined dataframe.

    Args:
        main_data (pd.DataFrame): Input dataframe containing samples and labels.
        train_val_split (float, optional): Proportion of data to assign to the
            training set. Must be between 0 and 1. Defaults to 0.9.
        csv_label_column (str): Name of the column containing class labels used
            for stratified sampling.
        random_state (int, optional): Random seed for reproducibility of the split.
            Defaults to 42.

    Returns:
        A dataframe containing both splits with an added dataset type ('train','validation') column
    """
    train_data, val_data = train_test_split(
        main_data, train_size=train_val_split, stratify=main_data[csv_label_column], random_state=random_state)
    train_data['dataset_type'] = 'train'
    val_data['dataset_type'] = 'validation'

    return pd.concat([train_data, val_data], ignore_index=True)
