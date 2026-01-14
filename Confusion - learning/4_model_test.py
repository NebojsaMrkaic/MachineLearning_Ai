import pytest
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from NNConfusion import ConfusionNet

@pytest.fixture
def prepared_data():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv("test_EEG_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")
    features = df.groupby(['subjectid', 'videoid']).agg({
        'theta': ['sum', 'mean', 'std', 'max', 'min'],
        'attention': 'mean',
        'mediation': 'mean'
    }).reset_index()
    features.columns = ['subjectid', 'videoid', 'theta_sum', 'theta_mean', 'theta_std', 'theta_max', 'theta_min', 'attention_mean', 'mediation_mean']
    features['confusion_level'] = features['videoid'].str.extract(r'(\d+)').astype(int)

    X = features[['theta_sum', 'theta_mean', 'theta_std', 'theta_max', 'theta_min', 'attention_mean', 'mediation_mean']].values.astype(np.float32)
    y = features['confusion_level'].values.astype(np.int64) - 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def test_model_training_and_prediction(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data

    model = ConfusionNet(input_dim=7)

    # 🔹 Balansiraj klase
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 🔹 Treniranje sa early stopping
    best_loss = float('inf')
    patience = 10
    trigger = 0

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger += 1
            if trigger >= patience:
                break

    # 🔹 Evaluacija
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test))
        predicted = torch.argmax(outputs, dim=1).numpy()
        accuracy = accuracy_score(y_test, predicted)

    # 🔹 Testovi
    assert accuracy > 0.6, f"Model accuracy too low: {accuracy:.2f}"
    assert len(set(predicted)) > 1, "Model predicts only one class"
    assert len(predicted) == len(y_test), "Prediction count mismatch"

    print(f"✅ Test passed with improved accuracy: {accuracy:.2f}")
