'''
Data utilities for loading and preprocessing training data.
Handles batch CSV files, path resolution across multiple mount points,
and train/validation splitting.
'''
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.device import is_main_process

# hardcoded paths for fix path
source_dirpath = '/mnt/auto-ekyc/idrecapture'
alt_source_dirpath = '/mnt3/auto-ekyc/idrecapture/datasets'
alt_alt_source_dirpath = '/mnt2/raw_dataset/research/idv/idfraud/data/'
live_source_dirpath = '/mnt/auto-ekyc/live_data/'

# Hardcoded paths for read_data (previously from ConfigManager)
MNT_PATH = "/mnt/auto-ekyc/idrecapture"
MNT2_PATH = "/mnt2/auto-ekyc/idrecapture"
PRIMARY_DATASET_PATH = "/mnt3/auto-ekyc/idrecapture/datasets"
SECONDARY_DATASET_PATH = "/mnt2/raw_dataset/research/idv/idfraud/data/"
LIVE_PATH = "/mnt/auto-ekyc/live_data/"

def fix_path(df,
             dataset_type,
             source_dirpath=source_dirpath,
             live_source_dirpath=live_source_dirpath,
             alt_source_dirpath=alt_source_dirpath,
             alt_alt_source_dirpath=alt_alt_source_dirpath,
             ):

    def check_path(x):
        path = os.path.join(alt_source_dirpath, x)  # check mnt 3 first
        if os.path.exists(path):
            return path
        path = os.path.join(source_dirpath, x)
        if os.path.exists(path):
            return path
        path = os.path.join(alt_alt_source_dirpath, "image_source", "batches", x) # check mnt 2 
        if os.path.exists(path):
            return path
        path = os.path.join(live_source_dirpath, x)
        if os.path.exists(path): # modified to raise an error and break training loop as soon as first not found image is present
            return path
        raise FileNotFoundError(f"Image not found: {x}")

    df = df[df['dataset_type'] == dataset_type]
    if is_main_process():
        tqdm.pandas(desc=f"Fixing paths ({dataset_type})")
        df['path'] = df['path'].progress_apply(check_path)
    else:
        df['path'] = df['path'].apply(check_path)

    return df

def read_data(image_type, batch_list, data_type, train_val_split=None, csv_image_column=None, split_data=True):

        def check_batch_duplicate(batch_list):
            """
            Method to check whether there are duplicate batches defined in configuration file (error if there is)
            """
            if len(batch_list) != len(set(batch_list)):
                print('There is duplicate batches inside the data configuration file')
                raise Exception('There is duplicate batches inside the data configuration file')

        # deleted legacy check_image_source function - refer to preprocessing_legacy.py

        def label_fraud_type(fraud_type):
            if fraud_type is None:
                raise ValueError("Fraud_type is None")
            elif fraud_type == 'genuine':
                return 0
            else:
                return 1

        def combine_batch(image_type, batch_list, data_type):
            """
            Method to combine batches together
            """
            if batch_list is None or len(batch_list) == 0:
                logging.info("Error in configuration - Batch list is empty.")
                print("Error in configuration - Batch list is empty.")
                raise Exception("Error in configuration - Batch list is empty.")
            batch_datas = []
            for idx in tqdm(range(len(batch_list)), desc="Processing batches", disable=not is_main_process()):
                batch_path = batch_list[idx]
                batch_name = os.path.join(*batch_path.split(os.sep)[:-1])
                csv_path = os.path.join(MNT2_PATH, data_type, batch_path)
                from_dataset = False

                if not os.path.exists(csv_path):
                    csv_path = os.path.join(PRIMARY_DATASET_PATH, batch_path)
                    if not os.path.exists(csv_path):
                        csv_path = os.path.join(SECONDARY_DATASET_PATH, 'image_source', 'batches', batch_path)
                        if not os.path.exists(csv_path):
                            csv_path = os.path.join(MNT_PATH, batch_path)
                    from_dataset = True

                if os.path.exists(csv_path):
                    batch_data = pd.read_csv(csv_path)
                    if csv_image_column:
                        if from_dataset:
                            batch_data['path'] = batch_data[csv_image_column].apply(lambda x: os.path.join(batch_name, x))
                        else:
                            batch_data['path'] = batch_data.apply(lambda x: os.path.join(x['batch_name'], x[csv_image_column]),axis=1)
                    else:
                        if from_dataset:
                            if image_type == 'crop':
                                batch_data['path'] = batch_data['ocr_path'].apply(lambda x: os.path.join(batch_name, x))
                            elif image_type == 'corner':
                                batch_data['path'] = batch_data['corner_path'].apply(lambda x: os.path.join(batch_name, x))
                            else:
                                batch_data['path'] = batch_data['ori_path'].apply(lambda x: os.path.join(batch_name, x))
                        else:
                            if image_type == 'crop':
                                batch_data['path'] = batch_data.apply(lambda x: os.path.join(x['batch_name'], x['ocr_path']),axis=1)
                            elif image_type == 'corner':
                                batch_data['path'] = batch_data.apply(lambda x: os.path.join(x['batch_name'], x['corner_path']),axis=1)
                            else:
                                batch_data['path'] = batch_data.apply(lambda x: os.path.join(x['batch_name'], x['ori_path']),axis=1)
                    batch_data['batch'] = batch_name
                    batch_data['filename'] = batch_data['path'].apply(os.path.basename)
                else:
                    raise FileNotFoundError("FileNotFoundError: " + csv_path)
                batch_datas.append(batch_data)

            main_data = pd.concat(batch_datas).copy()

            main_data['label'] = main_data['fraud_type'].apply(lambda x: 0 if x == 'genuine' else 1)
            grl = bool(int(os.environ.get('GRL', 0)))
            main_data = main_data.reset_index(drop=True)

            return main_data

        def split_data(main_data, train_val_split):
            train_data, val_data = train_test_split(
                main_data, train_size=train_val_split, stratify=main_data['label'], random_state=42)

            if train_data is not None:
                train_data['dataset_type'] = 'train'
            if val_data  is not None:
                val_data ['dataset_type'] = 'validation'
            return train_data, val_data

        check_batch_duplicate(batch_list)
        main_data = combine_batch(image_type, batch_list, data_type)

        if train_val_split:
            train_data, val_data = split_data(main_data, train_val_split)
            main_data = pd.concat([train_data, val_data], ignore_index=True)
            return main_data

        return main_data
