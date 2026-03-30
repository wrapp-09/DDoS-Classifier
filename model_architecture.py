import torch
import torch.nn as nn

class FeatureAutoencoder(nn.Module):
    """
    The Autoencoder:
    Compresses the 79 original features down to a 20-feature bottleneck
    to filter out network noise and capture the pure attack signature.
    """
    def __init__(self, input_dim=79, bottleneck_dim=20):
        super(FeatureAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return bottleneck, reconstruction


class BiLSTMClassifier(nn.Module):
    """
    The Classifier:
    Takes the compressed sequence from the Autoencoder and uses Bidirectional
    LSTMs to understand the chronological flow of the DDoS attack.
    """
    def __init__(self, input_dim=20, hidden_dim=64, num_layers=2, num_classes=3):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        final_time_step = lstm_out[:, -1, :]
        predictions = self.classifier(final_time_step)
        return predictions

if __name__ == "__main__":
    dummy_input = torch.rand((64, 10, 79))
    
    ae = FeatureAutoencoder()
    bilstm = BiLSTMClassifier()
    
    bottleneck, reconstructed = ae(dummy_input)
    print(f"Original shape: {dummy_input.shape}")
    print(f"Bottleneck shape: {bottleneck.shape}")
    
    predictions = bilstm(bottleneck)
    print(f"Final predictions shape: {predictions.shape}")
