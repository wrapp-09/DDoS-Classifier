# Temporal-Behavioral Classification of Zero-Day DDoS Attacks
**A Two-Phase AE-BiLSTM Architecture with a Mitigation-Focused Taxonomy**

This repository contains the code and methodology for an advanced machine learning pipeline designed to classify Zero-Day Distributed Denial of Service (DDoS) attacks. By shifting away from traditional, tool-specific classification (e.g., DNS, LDAP) and utilizing a **Mitigation-Focused Taxonomy** (Benign, Reflection, Exploitation), this model achieves **99.15% accuracy** on unseen threats.

## Core Architecture
Traditional firewalls evaluate packets as isolated events, which fails against complex, modern attacks. This project utilizes a temporal, 3D sliding-window approach:
1. **Phase 1 (FeatureAutoencoder):** An unsupervised `79 -> 32 -> 20` autoencoder that compresses network flow statistics to filter out environment-specific noise and extract pure behavioral signatures.
2. **Phase 2 (BiLSTM Classifier):** A supervised sequence model that analyzes the chronological flow of packets (Time Steps = 3) to accurately categorize the attack's super-class.

## Repository Structure
```text
DDoS-Classifier/
├── assets/                       # Generated training plots and confusion matrices
├── saved_models/                 # Stores trained .pth weights and JSON metrics
│   ├── autoencoder.pth
│   ├── bilstm_classifier.pth
│   └── training_metrics.json
├── data_transform.py             # Preprocessing, standardizing, and 3D windowing
├── model_architecture.py         # PyTorch definitions for AE and BiLSTM
├── train.py                      # Two-phase training loop
├── generate_plots.py             # Dynamically builds IEEE-ready graphs from JSON
├── stress_test.py                # Cross-dataset evaluation script (CIC-IDS-2017)
├── .gitignore                    
└── README.md                     
```

## Dataset & Setup
This project utilizes the **CICDDoS2019** dataset. *Note: Due to size constraints, the raw data files are not included in this repository.*

**1. Clone the repository:**
```bash
git clone [https://github.com/wrapp-09/DDoS-Classifier.git](https://github.com/wrapp-09/DDoS-Classifier.git)
cd DDoS-Classifier
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Prepare the Data:**
* Download the CSV files from the official [CICDDoS2019 dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html).
* Place the raw CSVs into a local `data/raw/` directory (make sure your `data/` folder is in `.gitignore`).
* Clean the data to remove whitespaces in column names, remove redundant columns, and to assign appropriate super-classes to the labels in each file.
```bash
python clean_data.py
```
* Run the preprocessing script balance classes via SMOTE, and generate the 3D numpy arrays:
```bash
python data_transform.py
```

## Usage

**Training the Model:**
Run the training script to execute the two-phase learning process. This will automatically save the best weights to the `saved_models/` folder and dump the metrics to a JSON file.
```bash
python train.py
```

**Generating Visualizations:**
Generate publication-ready Loss Curves and Confusion Matrices based on your latest training run:
```bash
python generate_plots.py
```

## Results
Evaluated on a strictly unseen Test Day distribution containing novel Zero-Day attacks (e.g., PortMap), the AE-BiLSTM achieved the following:
* **Overall Accuracy:** 99.15%
* **Macro F1-Score:** > 99.00%
* Successfully demonstrated that shrinking the temporal window (`time_steps=3`) dramatically improves the recall of burst-based Reflection attacks compared to standard MLP architectures.

## 🔬 Limitations & Future Work
While the model achieves near-perfect classification within the 2019 distribution, cross-dataset testing against CIC-IDS-2017 highlighted vulnerabilities to Feature Drift caused by updates in the `CICFlowMeter` extraction software. Future work will explore Domain Adversarial Neural Networks (DANN) to build entirely environment-agnostic models.
