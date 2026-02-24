import os
import json
import pandas as pd

from glob import glob

def jprint(dict):
    print(json.dumps(dict, indent=4))

debug = True

class ModelSelection:
    """ Model Selection to sort the model by weighted scoreboard. """
    def __init__(self, score_criteria, run_type, model_type=None):
        self.model_type = model_type
        self.score_weight = score_criteria['weight']
        self.lower_as_pass_criteria = score_criteria['lower_as_pass']
        self.greater_as_pass_criteria = score_criteria['greater_as_pass']

        # select_model_type = model_type if model_type != 'parallel' else 'ori'
        for criteria in [self.score_weight, self.lower_as_pass_criteria, self.greater_as_pass_criteria]:
            if criteria is not None:
                for field_data in criteria:
                    field_data['dataset'] = field_data['data_source']
                    field_data['field'] = field_data['data_source'] + '.' + field_data['metric']

        self.field_cols = []
        self.pass_criteria_field_cols = []
        for criteria in [self.score_weight]:
            for field_data in criteria:
                field = field_data['field']
                if field not in self.field_cols:
                    self.field_cols.append(field)
        for criteria in [self.lower_as_pass_criteria, self.greater_as_pass_criteria]:
            if criteria is not None:
                for field_data in criteria:
                    field = field_data['field']
                    if field not in self.field_cols:
                        self.field_cols.append(field)
                    if field not in self.pass_criteria_field_cols:
                        self.pass_criteria_field_cols.append(field)

        if run_type == 'idFraud':
            self.json_file = 'info.json'
        elif run_type == 'idPhysicalTamper':
            self.json_file = 'fraud_detection_summary.json'
        elif run_type == 'idLandmark':
            self.json_file = 'object_summary_report.json'

        self.bin_threshold = 0.5
    def find_filepath_by_basename(self, file_paths, target_basename):
        result = None
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            if os.path.splitext(filename)[0] == target_basename:
                result = file_path
        if result is None:
            return None
        return result
    

    def run_checkpoint(self, model, artifact_dir=None):
        model_path = os.path.join(artifact_dir, model)
        ckpt_paths = glob(os.path.join(model_path, 'eval/*/'))
        ckpt_model_paths = glob(os.path.join(model_path, 'checkpoints/*.pt'))
        return self.run(ckpt_paths, ckpt_model_paths)
    

    def run(self, ckpt_paths, ckpt_model_paths, thres_setting='all'):
        ckpt_dicts = self.get_ckpt_with_eval_dicts(ckpt_paths, ckpt_model_paths)
        leaderboard_df = self.generate_leaderboard(ckpt_dicts, thres_setting)
        
        # return the best model
        best_sorted_model = leaderboard_df.iloc[0,:].to_dict()
        selector = [True] * len(leaderboard_df)

        for criteria_standard, criteria in [['lower_as_pass', self.lower_as_pass_criteria], ['greater_as_pass', self.greater_as_pass_criteria]]:
            if criteria is not None:
                for field_data in criteria:
                    col = field_data['field'] + '_' + criteria_standard
                    selector &= leaderboard_df[col] == True
        best_sorted_model_with_pass = leaderboard_df[selector]
        
        if len(best_sorted_model_with_pass) > 0:
            best_sorted_model_with_pass = best_sorted_model_with_pass.iloc[0,:].to_dict()
        else:
            best_sorted_model_with_pass = None
        return best_sorted_model, best_sorted_model_with_pass, leaderboard_df
    
    
    def get_ckpt_with_eval_dicts(self, ckpt_paths, ckpt_model_paths):
        ckpt_dicts = [{'ckpt_path': p} for p in ckpt_paths]
        for i in range(len(ckpt_dicts)):
            ckpt_dict = ckpt_dicts[i]
            epoch = ckpt_dict['ckpt_path'].split('/')[-2]
            ckpt_dict['epoch'] = epoch
            ckpt_dict['ckpt_model_path'] = self.find_filepath_by_basename(ckpt_model_paths, epoch)
            # eval_paths = os.path.join(ckpt_dict['ckpt_path'], 'logs/evaluate/*/')
            # eval_paths = glob(eval_paths)
            eval_paths = glob(os.path.join(ckpt_dict['ckpt_path'],"*"))
            if debug:
                print('eval_paths', eval_paths)
                print('len(eval_paths)', len(eval_paths))
            ckpt_dict['evaluation'] = {}
            for eval_path in eval_paths:
                if debug:
                    print('eval_path', eval_path)
                eval_dict = {}
                eval_dict['eval_path'] = eval_path
                
                # search evaluation csv  
                dataset_name = eval_path.rsplit(os.sep, 1)[1]
                eval_dict['eval_csv'] = os.path.join(eval_path, f'{dataset_name}.csv')
                
                # read evaluation info json & get the metrics
                eval_dict['eval_info_json'] = os.path.join(eval_path, self.json_file)
                with open(eval_dict['eval_info_json']) as fid:
                    eval_info_json = json.loads(fid.read())
                eval_set = list(eval_info_json['metrics'].keys())
                eval_dict['threshold'] = self.bin_threshold 
                for key in eval_set:
                    eval_dict[key] = {}
                    metrics = eval_info_json['metrics'][key]
                    eval_dict[key]['TP'] = metrics['TP']
                    eval_dict[key]['FP'] = metrics['FP']
                    eval_dict[key]['TN'] = metrics['TN']
                    eval_dict[key]['FN'] = metrics['FN']
                    # eval_dict['accuracy'] = metrics['accuracy']
                    eval_dict[key]['apcer'] = max(0, metrics['apcer'])
                    eval_dict[key]['bpcer'] = max(0, metrics['bpcer'])
                    if 'acer' in metrics:
                        eval_dict[key]['acer'] = metrics['acer']
                    else:
                        eval_dict[key]['acer'] = (max(0, metrics['apcer']) + max(0, metrics['bpcer'])) /2
                    if 'accuracy' in metrics:
                        eval_dict[key]['accuracy'] = metrics['accuracy']
                    else:
                        eval_dict[key]['accuracy'] = metrics['acc']

                ckpt_dict['evaluation'][dataset_name] = eval_dict
        return ckpt_dicts  


        #         # read evaluation info json & get the metrics
        #         eval_dict['eval_info_json'] = os.path.join(eval_path, 'fraud_detection_summary.json')
        #         with open(eval_dict['eval_info_json']) as fid:
        #             eval_info_json = json.loads(fid.read())

        #         eval_dict['threshold'] = 0.5
        #         # eval_dict['threshold'] = eval_info_json['eval']['bin_threshold']

        #         key = 'threshold_0.5'
        #         eval_dict[key] = {}
        #         metrics = eval_info_json
        #         eval_dict[key]['TP'] = metrics['TP']
        #         eval_dict[key]['FP'] = metrics['FP']
        #         eval_dict[key]['TN'] = metrics['TN']
        #         eval_dict[key]['FN'] = metrics['FN']
        #         #eval_dict['accuracy'] = metrics['accuracy']
        #         eval_dict[key]['apcer'] = max(0, metrics['apcer'])
        #         eval_dict[key]['bpcer'] = max(0, metrics['bpcer'])
        #         if 'acer' in metrics:
        #             eval_dict[key]['acer'] = metrics['acer']
        #         else:
        #             eval_dict[key]['acer'] = (max(0, metrics['apcer']) + max(0, metrics['bpcer'])) /2
        #         if 'accuracy' in metrics:
        #             eval_dict[key]['accuracy'] = metrics['accuracy']
        #         else:
        #             eval_dict[key]['accuracy'] = metrics['acc']

        #         ckpt_dict['evaluation'][dataset_name] = eval_dict
        # return ckpt_dicts
    

    def generate_leaderboard(self, ckpt_dicts, thres_setting):  
        final_leaderboard_df = pd.DataFrame()
        if debug:
            print ("Ckpt Dicts: "+str(ckpt_dicts))            
        for ckpt_dict in ckpt_dicts: 
            if debug:
                print('ckpt_path', ckpt_dict['ckpt_path'])
            leaderboard_df_list = []  
            leaderboard_df_merged = pd.DataFrame()
            for field in self.field_cols: 
                if debug:
                    print('field', field)
                # Loop through declared field cols to extract required dataset name and metric for leaderboard.
                leaderboard_list = []
                dataset_name, metric = field.split('.')
                # dataset_name = dataset_name.rsplit('_',1)[0]
                ignore_keys = ['eval_path','eval_csv','eval_info_json','threshold'] # Keys to ignore in the ckpt_dict

                if dataset_name not in ckpt_dict['evaluation']:
                    raise ValueError(f"{dataset_name} not in checkpoint {ckpt_dict['ckpt_path']}")

                thresholds = ckpt_dict['evaluation'][dataset_name].keys()

                for key in thresholds:
                    # Loop through all threshold for the given dataset name and create a dataframe.
                    model_dict = {}
                    model_dict['model'] = ckpt_dict['ckpt_model_path']
                    model_dict['model_epoch'] = ckpt_dict['epoch'] 
                    if thres_setting == 'all' and key not in ignore_keys:
                        if self.model_type == 'parallel':
                            model_dict['threshold'] = key  # Create threshold column for "all" parallel model selection
                        
                        model_dict[field] = ckpt_dict['evaluation'][dataset_name][key][metric]
                        leaderboard_list.append(model_dict)
                        leaderboard_df = pd.DataFrame(leaderboard_list)

                    elif thres_setting == 'single' and key not in ignore_keys and key == 'threshold_0.5':
                        model_dict['threshold'] = key
                        model_dict[field] = ckpt_dict['evaluation'][dataset_name][key][metric]
                        leaderboard_list.append(model_dict)
                        leaderboard_df = pd.DataFrame(leaderboard_list)

                    elif thres_setting == 'single' and key not in ignore_keys and key == 'threshold' and ckpt_dict['evaluation'][dataset_name][key] == 0.5:
                        model_dict['threshold'] = 'threshold_0.5'
                        model_dict[field] = ckpt_dict['evaluation'][dataset_name]["test"][metric]
                        leaderboard_list.append(model_dict)
                        leaderboard_df = pd.DataFrame(leaderboard_list)

                leaderboard_df_list.append(leaderboard_df) # Append all the dataset with thresold datafrom into a single list.

            for index in range(len(leaderboard_df_list)):
                # Merge the list into a single dataframe.
                if leaderboard_df_merged.empty:
                    leaderboard_df_merged = leaderboard_df_list[index]
                else:
                    leaderboard_df_merged = pd.merge(leaderboard_df_merged, leaderboard_df_list[index])
            # Concat dataframes with one another when there are multiple models.
            final_leaderboard_df = pd.concat([final_leaderboard_df, leaderboard_df_merged], axis=0)
        leaderboard_df = final_leaderboard_df
        
        def apply_pass_criteria(row):
            """
            pass criteria x 2
                take every required score
                check criteria
            """    
            if self.lower_as_pass_criteria is not None:    
                for field_data in self.lower_as_pass_criteria:    
                    field = field_data['field']
                    pass_criteria = field_data['value']

                    metric_score = row[field]
                    row[f'{field}_lower_as_pass'] = metric_score < pass_criteria
            if self.greater_as_pass_criteria is not None:    
                for field_data in self.greater_as_pass_criteria:
                    field = field_data['field']
                    pass_criteria = field_data['value']
                    metric_score = row[field]
                    row[f'{field}_greater_as_pass'] = metric_score > pass_criteria
            return row
        leaderboard_df = leaderboard_df.apply(apply_pass_criteria, axis=1)
        def apply_score(row):
            """
            calculate the score 
                take every score weight
                get the value of the field
                multiply with the weight
                sum
            """    
            score = total_weight = 0
            for field_data in self.score_weight:
                
                field = field_data['field']
                weight = field_data['value']
                total_weight += abs(weight)
                metric_score = row[field]
                is_desc = weight < 0
                if is_desc:
                    metric_score = 1 - metric_score
                score += metric_score * abs(weight)
            score = score/total_weight
            return round(score, 3)
        if debug:
            print('leaderboard_df', leaderboard_df.columns)
        leaderboard_df['score'] = leaderboard_df.apply(apply_score, axis=1)
        leaderboard_df = leaderboard_df.sort_values(by='score', ascending=False)
        return leaderboard_df
