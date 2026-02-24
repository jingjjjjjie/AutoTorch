'''
Data utilities for loading and preprocessing training data.
Handles batch CSV files, path resolution across multiple mount points,
and train/validation splitting.
'''
import os
import logging
import pandas as pd
from tqdm import tqdm

# hardcoded paths for fix path
mnt3_path = '/mnt3/auto-ekyc/idrecapture/datasets'


def fix_path(df, source_dirpath=mnt3_path):
    errors = [] # list to store the paths that are not found in mnt 3

    # check if image is present in the source (mnt3)
    def check_path(x):
        path = os.path.join(source_dirpath, x)  # check mnt 3 only
        if os.path.exists(path):
            return path
        else:
            errors.append(x)  # accesses 'errors' from outer scope
            return None

    # apply the path fixing function to the dataframe
    tqdm.pandas(desc="Fixing paths")
    df['path'] = df['path'].progress_apply(check_path)

    # Filter out rows with None paths (unfixed)
    df_fixed = df[df['path'].notna()].copy()
    df_errors = df[df['path'].isna()].copy()

    # print warning if some paths are not found
    if errors:
        print(f"Warning: {len(errors)} paths are not found in mnt 3")
    return df_fixed, df_errors


def read_data(csv_image_column, batch_list):

    def check_batch_duplicate(batch_list):
        """Method to check whether there are duplicate batches"""
        if len(batch_list) != len(set(batch_list)):
            raise Exception('There is duplicate batches inside the data configuration file')

    def combine_batch(csv_image_column, batch_list):
        """
        Method to combine batches together
        """
        # first check if batch list is empty
        if batch_list is None or len(batch_list) == 0:
            logging.info("Error in configuration - Batch list is empty.")
            print("Error in configuration - Batch list is empty.")
            raise Exception("Error in configuration - Batch list is empty.")
            

        batch_datas = []
        batch_errors = []
        for idx in tqdm(range(len(batch_list)), desc="Processing batches"):
            batch_path = batch_list[idx]
            try:
                # batch_name ends up being the directory containing the file, with the filename (image.png) stripped off.
                # example batch_path would be: "batch_issue_20240704_snt_both_colorghostwhitebg/index_annotation_mykadfront.csv",
                # batch_name would be batch_issue_20240704_snt_both_colorghostwhitebg/
                batch_name = os.path.join(*batch_path.split(os.sep)[:-1])
                
                # join the path with the real location of the batch on the server
                csv_path = os.path.join(mnt3_path, batch_path)

                if os.path.exists(csv_path): # check if the csv path exists    
                    batch_data = pd.read_csv(csv_path)
                    # loop through each row (lambda x -> x is one row)
                    batch_data['path'] = batch_data[csv_image_column].apply(lambda x: os.path.join(batch_name, x)) # loop through each row, axis=1; 
                    batch_data['batch'] = batch_name
                    batch_data['filename'] = batch_data['path'].apply(os.path.basename)
                    batch_datas.append(batch_data)
                else:
                    batch_errors.append({"batch": batch_path, "error": f"CSV not found: {csv_path}"})
            except Exception as e:
                batch_errors.append({"batch": batch_path, "error": str(e)})

        if batch_errors:
            print(f"Warning: {len(batch_errors)} batches could not be loaded")
            for err in batch_errors:
                print(f"  - {err['batch']}: {err['error']}")

        if not batch_datas:
            raise Exception("No batches could be loaded")
        
        # concat all the seperate datas into one dataframe
        main_data = pd.concat(batch_datas).copy()

        # map the genuine and fraud labels to 0 and 1
        main_data['label'] = main_data['fraud_type'].apply(lambda x: 0 if x == 'genuine' else 1)
        
        # reset the index
        main_data = main_data.reset_index(drop=True)

        return main_data, batch_errors

    check_batch_duplicate(batch_list)
    main_data, batch_errors = combine_batch(csv_image_column, batch_list)
    
    return main_data, batch_errors



if __name__ == "__main__":
    # ============================================
    # Test read_data with a sample batch
    # ============================================
    print("=" * 50)
    print("Testing read_data()")
    print("=" * 50)

    # Example batch list - modify with your actual batch paths
    test_batch_list = [
        "batch_issue_20240704_snt_both_colorghodstwhitebg/index_annotation_mykadfront.csv",
        # Add more batches here
    ]

    try:
        main_data, errors = read_data(
            csv_image_column='ori_path',  # or 'ocr_path', 'corner_path'
            batch_list=test_batch_list
        )

        print(f"\nLoaded {len(main_data)} rows")
        print(f"Errors: {len(errors)}")
        print(f"\nColumns: {main_data.columns.tolist()}")
        print(f"\nSample data:")
        print(main_data[['path', 'batch', 'filename', 'label']].head())
        print(f"\nLabel distribution:")
        print(main_data['label'].value_counts())

    except Exception as e:
        print(f"Error: {e}")

    # ============================================
    # Test fix_path
    # ============================================
    print("\n" + "=" * 50)
    print("Testing fix_path()")
    print("=" * 50)

    # Use the data from read_data to test fix_path
    if 'main_data' in dir() and main_data is not None:
        print(f"Testing fix_path on {len(main_data)} rows...")
        df_fixed, df_errors = fix_path(main_data)
        print(f"Fixed: {len(df_fixed)} rows")
        print(f"Errors: {len(df_errors)} rows")
        if len(df_fixed) > 0:
            print(f"\nSample fixed paths:")
            print(df_fixed['path'].head())
