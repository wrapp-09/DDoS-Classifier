import os
import pandas as pd
import numpy as np

def build_master_dataset(input_folder, output_filename, attack_sample_size=50000):
    """
    Iterates through a folder of CICDDoS CSVs using chunking to prevent RAM overflow.
    Cleans the data, maps attacks to Super-Classes, and compiles a master dataset.
    """
    cols_to_drop = [
        'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
        'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP'
    ]
    
    master_list = []
    
    def map_attack_class(label):
        label = str(label).strip().upper()
        if label == 'BENIGN':
            return 'BENIGN'
        if any(kw in label for kw in ['MSSQL', 'SSDP', 'CHARGEN', 'NTP', 'TFTP', 'DNS', 'LDAP', 'NETBIOS', 'SNMP', 'PORTMAP']):
            return 'REFLECTION'
        if any(kw in label for kw in ['SYN', 'UDP']):
            return 'EXPLOITATION'
        return 'UNKNOWN'

    print(f"Scanning folder: {input_folder}...")
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f" -> Processing: {filename} (in chunks)")
            
            file_benign = []
            file_attack = []
            
            chunk_iterator = pd.read_csv(file_path, chunksize=250000, low_memory=False)
            
            for chunk in chunk_iterator:
                chunk.columns = chunk.columns.str.strip()
                chunk = chunk.drop(columns=[col for col in cols_to_drop if col in chunk.columns], errors='ignore')
                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.dropna(inplace=True)
                
                chunk['Label'] = chunk['Label'].apply(map_attack_class)
                chunk = chunk[chunk['Label'] != 'UNKNOWN']
                
                benign_chunk = chunk[chunk['Label'] == 'BENIGN']
                if not benign_chunk.empty:
                    file_benign.append(benign_chunk)
                
                attack_chunk = chunk[chunk['Label'] != 'BENIGN']
                if not attack_chunk.empty:
                    grab_size = min(len(attack_chunk), 5000) 
                    file_attack.append(attack_chunk.sample(n=grab_size, random_state=42))

            df_file_benign = pd.concat(file_benign) if file_benign else pd.DataFrame()
            df_file_attack = pd.concat(file_attack) if file_attack else pd.DataFrame()
            
            if len(df_file_attack) > attack_sample_size:
                df_file_attack = df_file_attack.sample(n=attack_sample_size, random_state=42)
                
            processed_file_df = pd.concat([df_file_benign, df_file_attack])
            if not processed_file_df.empty:
                master_list.append(processed_file_df)

    print("\nCompiling final dataset...")
    final_df = pd.concat(master_list, ignore_index=True)
    
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nSaved to {output_filename}.")
    print(f"Final Dataset Shape: {final_df.shape}")
    print("Class Distribution:")
    print(final_df['Label'].value_counts())
    print("-" * 40)

if __name__ == "__main__":
    build_master_dataset('.data/raw/Train_Day', './data/cleaned/master_train.csv')
    build_master_dataset('.data/raw/Test_Day', './data/cleaned/master_test.csv')
