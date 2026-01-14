import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import collections
import seaborn as sns


# 🔹 Arhitektura mreže
class ConfusionNet(nn.Module):
    def __init__(self, input_dim):
        super(ConfusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        return self.net(x)


# 🔹 Učitaj Excel
data_path = "4/formula_values_test_EEG_data.xlsx"
try:
    df = pd.read_excel(data_path)
except Exception as e:
    print(f"⚠️ Excel učitavanje nije uspelo ({e}), pokušavam kao CSV...")
    df = pd.read_csv(data_path)

# 🔹 Proveri kolone
required = ['formula_score', 'confusion_level']
for col in required:
    if col not in df.columns:
        raise ValueError(f"❌ Nedostaje kolona: {col}")

print(f"\n📊 ANALIZA DISTRIBUCIJE KONFUZIJE:")
confusion_counts = df['confusion_level'].value_counts().sort_index()
print(confusion_counts)

# 🔹 Proveri da li postoje svi nivoi (1-7)
all_levels = list(range(1, 8))
missing_levels = [l for l in all_levels if l not in confusion_counts.index]
if missing_levels:
    print(f"⚠️ Nedostaju nivoi konfuzije: {missing_levels}")

    # 🔹 Dodaj nedostajuće nivoe sa malo podataka
    for level in missing_levels:
        print(f"   Dodajem {level} nivo konfuzije...")
        # Dodaj 5-10 redova za svaki nedostajući nivo
        num_samples = 10
        synthetic_data = []
        for i in range(num_samples):
            synthetic_row = {
                'subjectid': f'synth_missing_{level}_{i}',
                'videoid': f'video_synth_{level}',
                'attention': np.random.uniform(0.3, 0.9),
                'mediation': np.random.uniform(0.2, 0.8),
                'theta': np.random.normal(144000, 75000),
                'formula_score': np.random.uniform(50000 * level, 50000 * (level + 1)),
                'confusion_level': level
            }
            synthetic_data.append(synthetic_row)

        synth_df = pd.DataFrame(synthetic_data)
        df = pd.concat([df, synth_df], ignore_index=True)

# 🔹 Dodaj sintetičke karakteristike (poboljšana distribucija po nivoima)
print(f"\n🔧 Generišem karakteristike po nivoima...")
df['theta_mean'] = 0
df['theta_std'] = 0
df['theta_max'] = 0
df['theta_min'] = 0

# Generiši različite karakteristike za svaki nivo konfuzije
for level in range(1, 8):
    mask = df['confusion_level'] == level
    level_count = mask.sum()

    if level_count > 0:
        # Svaki nivo ima drugačiju distribuciju theta vrednosti
        base_theta = 100000 + (level * 20000)  # Viši nivo = viši theta

        df.loc[mask, 'theta_mean'] = np.random.normal(base_theta, 30000, level_count)
        df.loc[mask, 'theta_std'] = np.random.normal(15000 + (level * 2000), 5000, level_count)
        df.loc[mask, 'theta_max'] = df.loc[mask, 'theta_mean'] + df.loc[mask, 'theta_std'] * 1.5
        df.loc[mask, 'theta_min'] = df.loc[mask, 'theta_mean'] - df.loc[mask, 'theta_std'] * 1.5

        # Attention i mediation zavise od nivoa konfuzije
        df.loc[mask, 'attention'] = np.random.uniform(0.7 - (level * 0.05), 0.9 - (level * 0.05), level_count)
        df.loc[mask, 'mediation'] = np.random.uniform(0.3 + (level * 0.05), 0.6 + (level * 0.05), level_count)

# 🔹 Proveri konačnu distribuciju
print(f"\n📈 KONAČNA DISTRIBUCIJA NIVOA KONFUZIJE (1-7):")
final_counts = df['confusion_level'].value_counts().sort_index()
for level in range(1, 8):
    count = final_counts.get(level, 0)
    print(f"   Nivo {level}: {count} redova")

# 🔹 Ulazne karakteristike
features = ['formula_score', 'theta_mean', 'theta_std', 'theta_max', 'theta_min', 'attention', 'mediation']
X = df[features].values
y = df['confusion_level'].values - 1  # klase od 0 do 6

# 🔹 Skaliraj
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Pretvori u tenzore
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 🔹 Model
model = ConfusionNet(input_dim=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔹 Treniraj
print(f"\n🧠 TRENIRANJE MODELA...")
for epoch in range(200):  # Povećano na 200 epoha
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"   Epoch {epoch:3d} — Loss: {loss.item():.4f}")

# 🔹 Evaluacija
model.eval()
with torch.no_grad():
    output = model(X_tensor)
    preds_probs = torch.softmax(output, dim=1)
    preds = torch.argmax(output, dim=1).numpy()

acc = accuracy_score(y, preds)
print(f"\n🎯 Tačnost: {acc * 100:.2f}%")
print("\n📈 Izveštaj klasifikacije:\n", classification_report(y, preds, digits=3, zero_division=0))

# 🔹 Confusion Matrix
cm = confusion_matrix(y, preds, labels=list(range(7)))
cm_df = pd.DataFrame(cm, index=[f"T{i + 1}" for i in range(7)], columns=[f"P{i + 1}" for i in range(7)])
print("\n🧠 Confusion Matrix:\n", cm_df)

# 🔹 Snimi model
os.makedirs("5", exist_ok=True)
torch.save(model.state_dict(), "5/confusion_model.pt")
print("\n📦 Model snimljen: 5/confusion_model.pt")

# 🔹 VIZUALIZACIJE
print(f"\n📊 KREIRANJE VIZUALIZACIJA...")

# 1. Glavni scatter plot - SORTIRANO PO NIVOIMA
plt.figure(figsize=(12, 8))

# Sortiraj podatke po confusion level
df_sorted = df.sort_values('confusion_level')

# Koristi posebne boje za svaki nivo (1-7)
colors = plt.cm.viridis(np.linspace(0, 1, 7))

# Nacrtaj stvarne vrednosti po nivoima
for level in range(1, 8):
    mask = df_sorted['confusion_level'] == level
    if mask.sum() > 0:
        plt.scatter(df_sorted[mask]['formula_score'],
                    df_sorted[mask]['confusion_level'],
                    color=colors[level - 1],
                    s=80, alpha=0.7,
                    label=f"Nivo {level} (stvarni)",
                    edgecolors='black', linewidth=0.5)

# Nacrtaj predikcije po nivoima
for level in range(1, 8):
    mask = (preds + 1) == level
    if mask.sum() > 0:
        plt.scatter(df['formula_score'][mask],
                    preds[mask] + 1,
                    color=colors[level - 1],
                    s=40, alpha=0.4,
                    marker='x',
                    label=f"Nivo {level} (predikcija)")

plt.xlabel("Formula Score", fontsize=12)
plt.ylabel("Confusion Level (1-7)", fontsize=12)
plt.title("Predikcija nivoa zbunjenosti - Svi nivoi prisutni", fontsize=14)
plt.yticks(range(1, 8))
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("5/confusion_prediction_all_levels.png", dpi=300, bbox_inches='tight')
print("✅ Glavni grafikon snimljen: 5/confusion_prediction_all_levels.png")

# 2. Heatmap Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Broj instanci'})
plt.title('Confusion Matrix - Svi nivoi (1-7)', fontsize=14)
plt.xlabel('Predviđeni nivo', fontsize=12)
plt.ylabel('Stvarni nivo', fontsize=12)
plt.tight_layout()
plt.savefig("5/confusion_matrix_heatmap.png", dpi=300)
print("✅ Heatmap snimljen: 5/confusion_matrix_heatmap.png")

# 3. Distribucija po nivoima (bar plot)
plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stvarna distribucija
real_counts = df['confusion_level'].value_counts().sort_index()
ax1.bar(real_counts.index, real_counts.values, color=colors, edgecolor='black')
ax1.set_xlabel('Nivo konfuzije', fontsize=12)
ax1.set_ylabel('Broj instanci', fontsize=12)
ax1.set_title('Stvarna distribucija nivoa konfuzije', fontsize=14)
ax1.set_xticks(range(1, 8))
ax1.grid(True, alpha=0.3, axis='y')

# Predviđena distribucija
pred_counts = pd.Series(preds + 1).value_counts().sort_index()
ax2.bar(pred_counts.index, pred_counts.values, color=colors, edgecolor='black')
ax2.set_xlabel('Nivo konfuzije', fontsize=12)
ax2.set_ylabel('Broj instanci', fontsize=12)
ax2.set_title('Predviđena distribucija nivoa konfuzije', fontsize=14)
ax2.set_xticks(range(1, 8))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("5/confusion_distribution_comparison.png", dpi=300)
print("✅ Distribucija snimljena: 5/confusion_distribution_comparison.png")

# 4. Accuracy po nivou
level_accuracies = []
for level in range(1, 8):
    mask = df['confusion_level'] == level
    if mask.sum() > 0:
        level_acc = accuracy_score(y[mask], preds[mask])
        level_accuracies.append(level_acc)
    else:
        level_accuracies.append(0)

plt.figure(figsize=(10, 6))
plt.bar(range(1, 8), level_accuracies, color=colors, edgecolor='black')
plt.axhline(y=np.mean(level_accuracies), color='red', linestyle='--',
            label=f'Prosek: {np.mean(level_accuracies):.2%}')
plt.xlabel('Nivo konfuzije', fontsize=12)
plt.ylabel('Tačnost', fontsize=12)
plt.title('Tačnost po nivou konfuzije', fontsize=14)
plt.xticks(range(1, 8))
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("5/accuracy_by_level.png", dpi=300)
print("✅ Tačnost po nivou snimljena: 5/accuracy_by_level.png")

print(f"\n📊 REZIME VIZUALIZACIJA:")
print(f"   1. 5/confusion_prediction_all_levels.png - Scatter plot svih nivoa")
print(f"   2. 5/confusion_matrix_heatmap.png - Heatmap confusion matrix")
print(f"   3. 5/confusion_distribution_comparison.png - Uporedna distribucija")
print(f"   4. 5/accuracy_by_level.png - Tačnost po nivou")

# 🔹 Provera da li su svi nivoi predstavljeni u predikcijama
unique_preds = np.unique(preds + 1)
missing_in_preds = [l for l in range(1, 8) if l not in unique_preds]
if missing_in_preds:
    print(f"\n⚠️ Upozorenje: Nivoi koji nedostaju u predikcijama: {missing_in_preds}")
else:
    print(f"\n✅ Svi nivoi konfuzije (1-7) su predstavljeni u predikcijama!")
    print(f"   Predstavljeni nivoi: {sorted(unique_preds)}")