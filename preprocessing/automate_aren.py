import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df_clean = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
    le = LabelEncoder()
    df_clean['Type'] = le.fit_transform(df_clean['Type'])
    
    X = df_clean.drop('Target', axis=1)
    y = df_clean['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('maintance_raw\maintance.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    os.makedirs('namadataset_preprocessing', exist_ok=True)
    pd.DataFrame(X_train).to_csv('namadataset_preprocessing/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('namadataset_preprocessing/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('namadataset_preprocessing/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('namadataset_preprocessing/y_test.csv', index=False)
    print("Preprocessing completed and saved.")