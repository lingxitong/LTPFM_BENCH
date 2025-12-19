# LTPFM_BENCH: Long-Tailed MIL Benchmark for Pathology Foundation Models

**LTPFM_BENCH** is a comprehensive PyTorch toolbox and benchmark suite designed for **Long-Tailed Multi-Instance Learning (MIL)** in Computational Pathology (CPath).

Beyond integrating state-of-the-art MIL backbones with advanced long-tailed strategies, LTPFM serves as a standardized benchmark for class-imbalanced WSI classification. We establish standardized data splitting protocols (based on Imbalance Ratios) and unified evaluation metrics  to enable systematic and consistent assessment of model performance on imbalanced pathological datasets.

## 📊 Long-Tailed Evaluation Protocol

To comprehensively evaluate performance, we categorize classes into **Many (Head)**, **Medium**, and **Few (Tail)** groups.

### Adaptive Fallback Mechanism

Unlike rigid thresholds (e.g., ImageNet-LT's <20), LTPFM employs an **adaptive strategy** to handle diverse medical datasets:

1. **Standard Protocol**:
   - **Many**: > 100 samples
   - **Few (Tail)**: < 20 samples
2. **Adaptive Fallback**:
   - If the dataset has no classes with < 20 samples, the toolbox automatically switches to a **Tertile-based Split**.
   - **Many (Head)** = The top 1/3 most frequent classes.
   - **Few (Tail)** = The bottom 1/3 least frequent classes.

## 🌟 Key Features

**1.Diverse MIL Methods:**

* **AB-MIL**: [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)
* **AMD-MIL**: [Agent Aggregator with Mask Denoise Mechanism](https://www.google.com/search?q=https://arxiv.org/abs/your_link&authuser=1)
* **CLAM-SB / MB**: [Data Efficient and Weakly Supervised Computational Pathology](https://arxiv.org/abs/2004.09666)
* **TransMIL**: [Transformer based Correlated Multiple Instance Learning](https://arxiv.org/abs/2106.00908)
* **WIKG-MIL**: [Dynamic Graph Representation with Knowledge-aware Attention](https://www.google.com/search?q=https://arxiv.org/abs/your_link&authuser=1)

**2.Long-Tailed Strategies:**

* **Re-sampling:** Class-balanced sampling.
* **Re-weighting:** Weighted Cross-Entropy,, Focal Loss, LDAM Loss.
* **Decoupling:** cRT (Classifier Re-training), LWS (Learnable Weight Scaling).
* **Ensemble:** BBN (Bilateral-Branch Network).
* **Logit Adjustment:** Balanced Softmax, T-Norm.
* UPDATING...

**3.Evaluation Metrics**

* **Acc /  AUC / F1-Score**
* **Head_Acc/Tail_Acc**
* **G-Mean**:Geometric mean of per-class sensitivity (Critical for imbalance).
* **Tail ECE**: Calibration error specifically computed on the **Few** group defined by the adaptive protocol.

## 📂 Data Preparation

Since WSI features are large, we expect users to extract features (using ResNet50, CTransPath, CONCH, PLIP, etc.) beforehand.

### 1. Directory Structure

Store your `.pt` feature files in a directory. Each file corresponds to one WSI.

```
DATA_ROOT/
└── features/
    ├── slide_A.pt
    ├── slide_B.pt
    └── ...
```

### 2. CSV Format

You need to prepare a `.csv` file with the following columns:

| slide_path               | label |
| :----------------------- | :---- |
| /path/to/your/slide_A.pt | 0     |
| /path/to/your/slide_B.pt | 1     |
| ...                      | ....  |

### 3. **Constructing Long-Tailed Datasets via IR**

For benchmarking purposes, LTPFM supports constructing synthetic long-tailed splits from original datasets. Specifically, we generate a **Long-Tailed Training Set** based on a specified Imbalance Ratio (IR) while maintaining a **Balanced Test Set** to ensure fair and unbiased evaluation.

```
python split_LT_Dataset.py \
                --csv_path /path/to/your/csv_file \
                --imbalance_ratio 50 \
```

## 🚀 Usage

### Training

You can train a model using `train_mil.py`. We use `hydra` and `argparse` for configuration management.

**Basic Example:** 

```bash
# Basic Example with CLAM-MB
python train_mil.py --yaml_path configs/CLAM_MB_MIL.yaml \
    --options Dataset.dataset_csv_path=datasets/your_LT_data.csv \
              General.num_epochs=50 \
              General.seed=2024
```

**Run with Long-Tailed Methods (e.g., LDAM):**

```bash
python train_mil.py --yaml_path configs/CLAM_MB_MIL.yaml \
    --options LT.method=LDAM \
              Model.in_dim=512
```

### **Automated Reporting**

We provide a script to batch-process experiment logs and generate a final academic report CSV.

```bash
python tools/analyze_results.py --path ./logs/
```



## 🔗 Acknowledgements

This code structure references and acknowledges the following repositories:

- **CLAM**: [https://github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)
- **MIL_BASELINE**:https://github.com/lingxitong/MIL_BASELINE
- **MONICA**:https://github.com/PyJulie/MONICA

