import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Class"

def load_data(path=r"C:\Programi_Python\Card fraud\data\creditcard.csv"):
    df = pd.read_csv(path)
    print("[INFO] Dataset loaded:", df.shape)
    print(df.head())
    return df

def preprocess(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Feature engineering: log-transform Amount
    if "Amount" in X.columns:
        X["Amount_log"] = np.log1p(X["Amount"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    print("[INFO] Train size:", X_train.shape, "Frauds:", sum(y_train))
    print("[INFO] Val size:", X_val.shape, "Frauds:", sum(y_val))
    print("[INFO] Test size:", X_test.shape, "Frauds:", sum(y_test))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

if __name__ == "__main__":
    df = load_data()
    preprocess(df)
