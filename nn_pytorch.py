import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_data import load_data

class AttritionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class AttritionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),    
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),   
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)
    
def train(epochs=50, batch_size=64, lr=0.001):
    X_train, X_test, y_train, y_test, scaler, feature_names = load_data()

    train_loader = DataLoader(
        AttritionDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True        # mélanger à chaque époque
    )
    test_loader = DataLoader(
        AttritionDataset(X_test, y_test),
        batch_size=batch_size
    )

    model     = AttritionNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()          
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # entrain
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()                
            optimizer.step()               
            train_loss += loss.item()

        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():              
            for X_batch, y_batch in test_loader:
                y_pred  = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
                correct  += ((y_pred >= 0.5) == y_batch).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(test_loader)
        accuracy  = correct    / len(y_test)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train Loss: {avg_train:.4f} | '
                  f'Val Loss: {avg_val:.4f} | '
                  f'Val Acc: {accuracy:.2%}')

    torch.save(model.state_dict(), 'model_pytorch.pth')
    import joblib
    joblib.dump(scaler,        'scaler_nn.pkl')
    joblib.dump(feature_names, 'feature_names_nn.pkl')
    torch.save(torch.tensor([X_train.shape[1]]), 'input_dim.pt')

    print('\n✓ Modèle PyTorch sauvegardé')
    return model, history

if __name__ == '__main__':
    model, history = train()


def predict_from_csv(csv_path, model_path='model_pytorch.pth'):
    import joblib

    feature_names = joblib.load('feature_names_nn.pkl')
    scaler        = joblib.load('scaler_nn.pkl')
    input_dim     = torch.load('input_dim.pt').item()

    df = pd.read_csv(csv_path)

    ids = df['EmployeeID'].values if 'EmployeeID' in df.columns \
          else np.arange(len(df))

    for col in ['EmployeeCount', 'Over18', 'StandardHours',
                'EmployeeID', 'Attrition']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    for col in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]  

    X = scaler.transform(df.values)

    model = AttritionNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        probas   = model(X_tensor).numpy()

    results = pd.DataFrame({
        'EmployeeID'      : ids,
        'Proba_Attrition' : (probas * 100).round(1),
        'Risque'          : ['🔴 Élevé'  if p >= 0.6 else
                             '🟡 Modéré' if p >= 0.35 else
                             '🟢 Faible'  for p in probas]
    }).sort_values('Proba_Attrition', ascending=False)

    print(results.to_string(index=False))
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        predict_from_csv(sys.argv[1])
    else:
        model, history = train()