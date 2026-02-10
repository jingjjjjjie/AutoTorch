import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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
             alt_source_dirpath = alt_source_dirpath,
             alt_alt_source_dirpath= alt_alt_source_dirpath,
             mnt2_path=MNT2_PATH):

    def check_path(x):
        path = os.path.join(source_dirpath, x)
        if not os.path.exists(path):
            path = os.path.join(alt_source_dirpath, x)
        if not os.path.exists(path):
            path = os.path.join(alt_alt_source_dirpath, "image_source", "batches", x)
        if not os.path.exists(path):
            path = os.path.join(live_source_dirpath, x)
        return path

    df = df[df['dataset_type'] == dataset_type]
    tqdm.pandas(desc=f"Fixing paths ({dataset_type})")
    df['path'] = df['path'].progress_apply(check_path)

    return df

def read_data(image_type, batch_list, data_type, train_val_split = None, csv_image_column=None):

        def check_batch_duplicate(batch_list):
            """
            Method to check whether there are duplicate batches defined in configuration file (error if there is)
            """
            if len(batch_list) != len(set(batch_list)):
                print('There is duplicate batches inside the data configuration file')
                raise Exception('There is duplicate batches inside the data configuration file')

        def check_image_source(main_data):
            """
            Method to check whether image source is present in corresponding folder
            """
            # config = configManager.load_idrecapture_config()
            image_absence = False
            for row in tqdm(range(len(main_data)), desc="Checking image sources"):
                image_path = os.path.join(MNT_PATH, main_data.loc[row]['path'])
                # print("Mnt: ", image_path)
                if not os.path.exists(image_path):
                    # image_path = os.path.join(config['root_path']['dataset_path'], 'image_source', 'batches', main_data.loc[row]['path'])
                    image_path = os.path.join(PRIMARY_DATASET_PATH, main_data.loc[row]['path'])
                    # print("Primary: ", image_path)
                if not os.path.exists(image_path):
                    image_path = os.path.join(SECONDARY_DATASET_PATH, 'image_source', 'batches', main_data.loc[row]['path'])
                    # print("Secondary: ", image_path)
                if not os.path.exists(image_path):
                    image_path = os.path.join(LIVE_PATH, main_data.loc[row]['path'])
                    # print("Live: ", image_path)

                # if not os.path.exists(os.path.join(os.path.dirname(self.source_dir), 'batches', main_data.loc[row]['path'])):
                if not os.path.exists(image_path):
                    print(f'Image source with file path {image_path} does not exist')
                    image_absence = True
            if image_absence:
                logging.info(
                    "Error in image data - some image filepaths within the main data is not available (check logs for more information)")
                print("Error in image data - some image filepaths within the main data is not available (check logs for more information)")
                raise Exception("Error in image data - some image filepaths within the main data is not available (check logs for more information)")

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
            # config = configManager.load_idrecapture_config()
            if batch_list is None or len(batch_list) == 0: # To check whether the batch list acquire is empty
                logging.info("Error in configuration - Batch list is empty.")
                print("Error in configuration - Batch list is empty.")
                raise Exception("Error in configuration - Batch list is empty.")
            batch_datas = []
            for idx in tqdm(range(len(batch_list)), desc="Processing batches"):
                batch_path = batch_list[idx]
                # batch_name = batch_path.split(os.sep)[0]
                batch_name = os.path.join(*batch_path.split(os.sep)[:-1])
                csv_path = os.path.join(MNT2_PATH, data_type, batch_path)
                from_dataset = False

                if not os.path.exists(csv_path):
                    # csv_path = os.path.join(config['root_path']['dataset_path'], 'image_source', 'batches', batch_path)
                    csv_path = os.path.join(PRIMARY_DATASET_PATH, batch_path)
                    if not os.path.exists(csv_path):
                        csv_path = os.path.join(SECONDARY_DATASET_PATH, 'image_source', 'batches', batch_path)
                        if not os.path.exists(csv_path):
                            csv_path = os.path.join(MNT_PATH, batch_path)
                    from_dataset = True

                if os.path.exists(csv_path):
                    batch_data = pd.read_csv(csv_path)
                    # if 'path' not in batch_data:
                    if csv_image_column:
                        if from_dataset:
                            batch_data['path'] = batch_data[csv_image_column].apply(lambda x: os.path.join(batch_name, x))
                        else:
                            batch_data['path'] = batch_data.apply(lambda x: os.path.join(x['batch_name'], x[csv_image_column]),axis=1)
                    else: 
                        if from_dataset:
                            if image_type == 'crop':
                                print(batch_data.columns)
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
            print("Concatenating batches...")
            main_data = pd.concat(batch_datas).copy()
            print(f"Concatenated {len(main_data)} rows. Adding labels...")
            main_data['label'] = main_data['fraud_type'].apply(lambda x: 0 if x == 'genuine' else 1)
            grl = bool(int(os.environ.get('GRL', 0)))
            # main_data['label'] = main_data['fraud_type'].apply(lambda x: label_fraud_type(x))
            print("Resetting index...")
            main_data = main_data.reset_index(drop=True)
            print("Data processing complete.")
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
        # Random subsample
        # main_data = main_data.sample(frac=0.05)
        if data_type == 'train':
            train_data, val_data = split_data(main_data, train_val_split)
            main_data = pd.concat([train_data, val_data], ignore_index=True)
        check_image_source(main_data)
        return main_data

def visualize_sample_image_from_dataloader(dataloader):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    true_img = None
    false_img = None

    # Loop until we find both classes
    for images, labels in dataloader:
        for i in range(len(labels)):
            if labels[i] == 1 and true_img is None:
                true_img = images[i]

            if labels[i] == 0 and false_img is None:
                false_img = images[i]

            if true_img is not None and false_img is not None:
                break
        if true_img is not None and false_img is not None:
            break

    def prep(img):
        img = img * std + mean
        img = img.permute(1,2,0).numpy()
        return img.clip(0,1)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(prep(false_img))
    plt.title("Genuine (0)")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(prep(true_img))
    plt.title("Fraud (1)")
    plt.axis('off')

    plt.show()