import os
import pandas as pd
import numpy as np
from scipy.stats import gmean
import glob
import yaml
import argparse
import warnings


warnings.filterwarnings('ignore')

DEFAULT_DATASET_ROOT = "default_path"
SAVE_PATH= "your_path/Final_Academic_Report.csv"
THRESH_MANY = 100  # > 100
THRESH_FEW = 20    # < 20


def parse_confusion_matrix(cm_str):
    try:
        cm_str = cm_str.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
        cm_str = cm_str.replace('\n', ' ')
        values = [int(x) for x in cm_str.split() if x.isdigit()]
        if not values: return None
        size = int(np.sqrt(len(values)))
        if size * size != len(values): return None
        return np.array(values).reshape(size, size)
    except:
        return None

def get_col_value(row, col_name):
    if col_name in row and pd.notna(row[col_name]):
        return float(row[col_name])
    return np.nan

def get_cls_distribution(model_dir):
 
    yaml_files = glob.glob(os.path.join(model_dir, "**/*.yaml"), recursive=True)
    csv_path = None
    for yf in yaml_files:
        if "hydra" in yf: continue
        try:
            with open(yf, 'r') as f:
                cfg = yaml.safe_load(f)
                if 'Dataset' in cfg and 'dataset_csv_path' in cfg['Dataset']:
                    csv_path = cfg['Dataset']['dataset_csv_path']
                    break
                if 'dataset_csv_path' in cfg:
                    csv_path = cfg['dataset_csv_path']
                    break
        except: continue

    try:
        df = pd.read_csv(csv_path)
        col = 'label' if 'label' in df.columns else df.columns[1]
        return df[col].value_counts().sort_index().tolist()
    except: return None

# def get_split_indices(cls_num_list):
#     many, med, few = [], [], []
#     for i, c in enumerate(cls_num_list):
#         if c > THRESH_MANY: many.append(i)
#         elif c < THRESH_FEW: few.append(i)
#         else: med.append(i)
#     return many, med, few

def get_split_indices(cls_num_list):

    if not isinstance(cls_num_list, list):
        cls_num_list = list(cls_num_list)
        
    many, med, few = [], [], []

    for i, c in enumerate(cls_num_list):
        if c > THRESH_MANY: many.append(i)
        elif c < THRESH_FEW: few.append(i)
        else: med.append(i)
        
    if len(few) == 0 or len(many) == 0:
        num_classes = len(cls_num_list)
        sorted_indices = np.argsort(cls_num_list)[::-1]
        split_1 = int(np.ceil(num_classes / 3))     
        split_2 = int(np.ceil(num_classes * 2 / 3)) 
        
        if num_classes < 3:
            many = sorted_indices[:1].tolist()   
            few  = sorted_indices[-1:].tolist() 
            med  = []
        else:
            many = sorted_indices[:split_1].tolist()
            med  = sorted_indices[split_1:split_2].tolist()
            few  = sorted_indices[split_2:].tolist()
            
    return many, med, few


def calculate_metrics(log_path, splits):
    try:
        df = pd.read_csv(log_path)
        if df.empty: return None
        row = df.iloc[-1] 
        res = {
            'Acc': get_col_value(row, 'test_acc'),
            'AUC': get_col_value(row, 'test_macro_auc'),
            'F1':  get_col_value(row, 'test_macro_f1'),
            'Tail ECE':get_col_value(row,'test_tail_ece')
        }

        if 'test_confusion_mat' in row:
            cm = parse_confusion_matrix(row['test_confusion_mat'])
            if cm is not None:
                row_sums = cm.sum(axis=1); row_sums[row_sums==0]=1
                acc_per_cls = np.diag(cm)/row_sums
                
                many_idx, med_idx, few_idx = splits
                res['G-Mean'] = gmean(acc_per_cls + 1e-8)
                res['Many'] = np.mean(acc_per_cls[many_idx]) if many_idx else np.nan
                res['Med']  = np.mean(acc_per_cls[med_idx])  if med_idx  else np.nan
                res['Few']  = np.mean(acc_per_cls[few_idx])  if few_idx  else np.nan
            else:
                res.update({'G-Mean':np.nan, 'Many':np.nan, 'Med':np.nan, 'Few':np.nan})
        else:
            res.update({'G-Mean':np.nan, 'Many':np.nan, 'Med':np.nan, 'Few':np.nan})
            
        return res
    except: return None

def analyze_group(model_dir, method, dataset):

    dist = get_cls_distribution(model_dir)
    if not dist: return None
    splits = get_split_indices(dist)


    csvs = glob.glob(os.path.join(model_dir, "**", "Log_*.csv"), recursive=True)
    if not csvs: return None

    keys = ['Acc', 'AUC', 'F1', 'G-Mean', 'Many', 'Med', 'Few', 'Tail ECE']
    agg = {k:[] for k in keys}
    
    valid = 0
    for f in csvs:
        m = calculate_metrics(f, splits)
        if m:
            valid += 1
            for k,v in m.items():
                if not np.isnan(v): agg[k].append(v)
    
    if valid == 0: return None

    row = {'Method': method, 'Dataset': dataset, 'Model': os.path.basename(model_dir), 'N': valid}
    for k, v_list in agg.items():
        if v_list: row[k] = f"{np.mean(v_list):.4f}±{np.std(v_list):.4f}"
        else: row[k] = "-"
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to logs root')
    args = parser.parse_args()

    print(f"🚀 Scanning: {args.path}")
    all_logs = glob.glob(os.path.join(args.path, "**", "Log_*.csv"), recursive=True)
    model_dirs = sorted(list(set([os.path.dirname(os.path.dirname(f)) for f in all_logs])))
    
    results = []
    for md in model_dirs:
        method = os.path.basename(os.path.dirname(md)) 
        dataset = os.path.basename(os.path.dirname(os.path.dirname(md)))    
        res = analyze_group(md, method, dataset)
        if res: results.append(res)

    if results:
        df = pd.DataFrame(results)
        cols = ['Dataset', 'Method', 'Model', 'N', 'Acc', 'AUC', 'F1', 'G-Mean', 'Many', 'Med', 'Few', 'Tail ECE']
        df = df[[c for c in cols if c in df.columns]]
        print("\n" + "="*120)
        print(df.to_string(index=False))
        print("="*120)
        df.to_csv(SAVE_PATH, index=False)
        print("\n✅ Saved to Final_Academic_Report.csv")
    else:
        print("❌ No valid experiments found.")

if __name__ == "__main__":
    main()