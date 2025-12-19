import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter

def make_balanced_test(X, y, seed):
    """
    Test set peaking logic: find the class with the lowest number and randomly cut the other classes to that number
    """
    df = pd.concat([X, y], axis=1)
    label_col = y.name
    

    counts = df[label_col].value_counts()
    min_samples = counts.min() 
    
    print(f"[Test Set] Clipping all classes to {min_samples} samples (Balanced).")
    
    balanced_df = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(min_samples, random_state=seed)
    )
    
    return balanced_df.drop(label_col, axis=1), balanced_df[label_col]

def make_long_tailed_train(X, y, imbalance_ratio, seed):
    """
    Training set long-tail logic: sorting by number, applying exponential decay formula for sampling
    """
    if imbalance_ratio == 1:
        return X, y
        
    df = pd.concat([X, y], axis=1)
    label_col = y.name
    

    counts = df[label_col].value_counts() 
    sorted_classes = counts.index.tolist()
    
    n_max = counts.max() 
    num_classes = len(sorted_classes)
    
    print(f"[Train Set] Generating Long-Tailed distribution with IR={imbalance_ratio}...")
    print(f"   - Head Class Count (Rank 0): {n_max}")
    
    final_indices = []
    
    for rank, cls in enumerate(sorted_classes):
        original_count = counts[cls]
        decay_factor = (1.0 / imbalance_ratio) ** (rank / (num_classes - 1.0))
        target_num = int(n_max * decay_factor)
        keep_num = min(target_num, original_count)        
        cls_indices = df[df[label_col] == cls].index
        np.random.seed(seed)
        selected_indices = np.random.choice(cls_indices, keep_num, replace=False)
        final_indices.extend(selected_indices)
        

    long_tailed_df = df.loc[final_indices]
    

    tail_cls = sorted_classes[-1]
    print(f"   - Tail Class Count (Rank {num_classes-1}): Target {int(n_max/imbalance_ratio)} -> Actual {len(long_tailed_df[long_tailed_df[label_col]==tail_cls])}")

    return long_tailed_df.drop(label_col, axis=1), long_tailed_df[label_col]

def LongTail_Train_Balanced_Test(args):
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)


    X = df.drop('label', axis=1)  
    y = df['label']

    train_size = args.train_ratio 
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, 
        train_size=train_size, 
        stratify=y, 
        random_state=args.seed, 
        shuffle=True
    )


    X_test_bal, y_test_bal = make_balanced_test(X_test_raw, y_test_raw, args.seed)

    X_train_lt, y_train_lt = make_long_tailed_train(X_train_raw, y_train_raw, args.imbalance_ratio, args.seed)

    
    train_df = pd.DataFrame({
        'train_slide_path': X_train_lt.values.flatten(), 
        'train_label': y_train_lt.values
    }).reset_index(drop=True)
    
    test_df = pd.DataFrame({
        'test_slide_path': X_test_bal.values.flatten(),
        'test_label': y_test_bal.values
    }).reset_index(drop=True)


    val_df = pd.DataFrame({
        'val_slide_path': [],
        'val_label': []
    })


    result = pd.concat([train_df, val_df, test_df], axis=1)

    print(f"Saving split result to {args.save_path}")
    result.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV')
    argparser.add_argument('--save_path', type=str, default='./split_split.csv')
    argparser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data used for Training pool')
    argparser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of data used for Test pool')
    

    argparser.add_argument('--imbalance_ratio', type=int, default=50, help='Imbalance Ratio (IR) for generating long-tailed training set')
    
    args = argparser.parse_args()
    

    assert abs((args.train_ratio + args.test_ratio) - 1.0) < 1e-9 , 'train_ratio + test_ratio must be equal to 1'
    
    LongTail_Train_Balanced_Test(args)