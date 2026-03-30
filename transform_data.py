import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import gc

def create_sequences(features, labels, time_steps):
    """
    Slides a window across 2D data to create 3D sequences for the BiLSTM.
    """
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i : (i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

def process_and_transform(time_steps=10, target_samples=100000):
    print("Loading datasets...")
    train_df = pd.read_csv('./data/cleaned/master_train.csv')
    test_df = pd.read_csv('./data/cleaned/master_test.csv')

    X_train = train_df.drop('Label', axis=1).values
    y_train_text = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test_text = test_df['Label'].values
    
    del train_df, test_df
    gc.collect()

    print("Encoding text labels to integers...")
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    y_test = encoder.transform(y_test_text)
    
    class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"Class Mapping: {class_mapping}")

    print("Balancing the Training Data (Hybrid Undersample + SMOTE)...")
    strategy_under = {class_mapping['REFLECTION']: target_samples}
    strategy_over = {class_mapping['EXPLOITATION']: target_samples, class_mapping['BENIGN']: target_samples}
    
    under = RandomUnderSampler(sampling_strategy=strategy_under, random_state=42)
    over = SMOTE(sampling_strategy=strategy_over, random_state=42)
    pipeline = Pipeline(steps=[('u', under), ('o', over)])
    
    X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)
    
    print(f"Balanced Train Shape: {X_train_bal.shape}")
    
    print("Scaling features (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    print(f"Creating 3D Sliding Windows (Time Steps = {time_steps})...")
    X_train_3d, y_train_seq = create_sequences(X_train_scaled, y_train_bal, time_steps)
    X_test_3d, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

    print(f"Final 3D Train Shape: {X_train_3d.shape}")
    print(f"Final 3D Test Shape:  {X_test_3d.shape}")

    print("Saving to disk as compressed Numpy arrays...")
    np.save('./data/processed/X_train_3d.npy', X_train_3d)
    np.save('./data/processed/y_train.npy', y_train_seq)
    np.save('./data/processed/X_test_3d.npy', X_test_3d)
    np.save('./data/processed/y_test.npy', y_test_seq)
    
    print("Data is ready for the AE-BiLSTM")

if __name__ == "__main__":
    process_and_transform(time_steps=3, target_samples=100000)
