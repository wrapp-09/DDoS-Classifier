import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def load_metrics(filepath='./saved_models/training_metrics.json'):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run train.py first!")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_curves(metrics):
    ae_losses = metrics['ae_losses']
    clf_losses = metrics['clf_losses']
    
    epochs_ae = list(range(1, len(ae_losses) + 1))
    epochs_lstm = list(range(1, len(clf_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_ae, ae_losses, marker='o', color='#1f77b4', linewidth=2)
    ax1.set_title('Phase 1: Autoencoder Reconstruction Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_xticks(epochs_ae)

    ax2.plot(epochs_lstm, clf_losses, marker='s', color='#ff7f0e', linewidth=2)
    ax2.set_title('Phase 2: BiLSTM Classifier Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_xticks(epochs_lstm)

    plt.tight_layout()
    plt.savefig('assets/Training_Loss_Curves.png', dpi=300)
    print("Saved'Training_Loss_Curves.png'")

def plot_confusion_matrix(metrics):
    cm = np.array(metrics['confusion_matrix'])
    
    if cm.shape[0] == 3:
        labels = ['Benign', 'Exploitation', 'Reflection']
    else:
        labels = ['Benign', 'Exploitation']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    
    plt.title('AE-BiLSTM Confusion Matrix', pad=20, fontweight='bold', fontsize=14)
    plt.ylabel('True Attack Super-Class', fontweight='bold')
    plt.xlabel('Predicted Attack Super-Class', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/Confusion_Matrix.png', dpi=300)
    print("Saved 'Confusion_Matrix.png'")

if __name__ == "__main__":
    data = load_metrics()
    if data:
        plot_training_curves(data)
        plot_confusion_matrix(data)
