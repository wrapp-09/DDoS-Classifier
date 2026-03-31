import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

from model_architecture import FeatureAutoencoder, BiLSTMClassifier

def load_data(batch_size=256):
    print("Loading datasets...")
    X_train = np.load('./data/processed/X_train_3d.npy')
    y_train = np.load('./data/processed/y_train.npy')
    X_test = np.load('./data/processed/X_test_3d.npy')
    y_test = np.load('./data/processed/y_test.npy')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on: {device}")

    autoencoder = FeatureAutoencoder().to(device)
    classifier = BiLSTMClassifier().to(device)

    train_loader, test_loader = load_data(batch_size=256)
    
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    clf_criterion = nn.CrossEntropyLoss()
    clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-5)

    # ==========================================
    # Autoencoder Training
    # ==========================================
    ae_epochs = 10
    ae_loss_history = []
    print("\nTraining Autoencoder (Feature Extraction)...")
    for epoch in range(ae_epochs):
        autoencoder.train()
        total_loss = 0
        
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            
            ae_optimizer.zero_grad()
            _, reconstructed = autoencoder(batch_X)
            
            loss = ae_criterion(reconstructed, batch_X)
            loss.backward()
            ae_optimizer.step()
            
            total_loss += loss.item()
        
        epoch_loss = total_loss/len(train_loader)
        ae_loss_history.append(epoch_loss)
            
        print(f"AE Epoch [{epoch+1}/{ae_epochs}] - Reconstruction Loss: {epoch_loss:.4f}")

    # ==========================================
    # BiLSTM Classifier Training
    # ==========================================
    clf_epochs = 6
    clf_loss_history = []
    print("\nTraining BiLSTM Classifier...")
    autoencoder.eval() 
    
    for epoch in range(clf_epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            clf_optimizer.zero_grad()
            
            with torch.no_grad():
                bottleneck, _ = autoencoder(batch_X)
            
            outputs = classifier(bottleneck)
            loss = clf_criterion(outputs, batch_y)
            
            loss.backward()
            clf_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = total_loss/len(train_loader)
        clf_loss_history.append(epoch_loss)    
        acc = 100 * correct / total
        print(f"BiLSTM Epoch [{epoch+1}/{clf_epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {acc:.2f}%")

    # ==========================================
    # Evaluation on Unseen Test Data
    # ==========================================
    print("\nEvaluating on Unseen Test Data...")
    classifier.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            
            bottleneck, _ = autoencoder(batch_X)
            outputs = classifier(bottleneck)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    print("\n" + "="*50)
    print("FINAL CLASSIFICATION REPORT")
    print("="*50)
    
    target_names = ['BENIGN', 'EXPLOITATION', 'REFLECTION']
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    
    final_acc = accuracy_score(all_targets, all_preds) * 100
    print(f"Overall Test Accuracy: {final_acc:.2f}%")

    torch.save(autoencoder.state_dict(), './saved_models/autoencoder.pth')
    torch.save(classifier.state_dict(), './saved_models/bilstm_classifier.pth')
    print("Model weights successfully saved")

    cm = confusion_matrix(all_targets, all_preds)
    metrics_dict = {
        "ae_losses": ae_loss_history,
        "clf_losses": clf_loss_history,
        "confusion_matrix": cm.tolist()
    }

    with open('./saved_models/training_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print("Metrics successfully saved")

if __name__ == "__main__":
    train_model()
