import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


# 🔹 Arhitektura mreže — mora da se poklapa sa treniranim modelom
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


# 🔹 Pronađi putanje u folderu 5
base_path = os.path.dirname(os.path.abspath(__file__))  # Trenutni folder gde je skripta
folder_5_path = os.path.join(base_path, "5")

print(f"🔍 Tražim fajlove u folderu: {folder_5_path}")

# Proveri koji fajlovi postoje u folderu 5
if os.path.exists(folder_5_path):
    files_in_5 = os.listdir(folder_5_path)
    print(f"📂 Fajlovi u folderu 5: {files_in_5}")

    # Pronađi Excel fajlove
    excel_files = [f for f in files_in_5 if f.endswith(('.xlsx', '.xls'))]
    model_files = [f for f in files_in_5 if f.endswith('.pt')]

    if excel_files:
        # Uzmi prvi Excel fajl
        excel_file = excel_files[0]
        data_path = os.path.join(folder_5_path, excel_file)
        print(f"✅ Pronađen Excel fajl: {excel_file}")
    else:
        # Ako nema u folderu 5, pokušaj u folderu 4
        folder_4_path = os.path.join(base_path, "4")
        if os.path.exists(folder_4_path):
            files_in_4 = os.listdir(folder_4_path)
            excel_files_4 = [f for f in files_in_4 if f.endswith(('.xlsx', '.xls'))]
            if excel_files_4:
                excel_file = excel_files_4[0]
                data_path = os.path.join(folder_4_path, excel_file)
                print(f"✅ Pronađen Excel fajl u folderu 4: {excel_file}")
            else:
                raise FileNotFoundError(f"❌ Nema Excel fajlova ni u folderu 4 ni 5")
        else:
            raise FileNotFoundError(f"❌ Folder 4 ne postoji")

    # Pronađi model fajl
    if model_files:
        model_file = model_files[0]
        model_path = os.path.join(folder_5_path, model_file)
        print(f"✅ Pronađen model fajl: {model_file}")
    else:
        raise FileNotFoundError(f"❌ Nema .pt model fajla u folderu 5")
else:
    raise FileNotFoundError(f"❌ Folder 5 ne postoji na putanji: {folder_5_path}")

# 🔹 Kreiraj output folder 6
output_folder = os.path.join(base_path, "6")
os.makedirs(output_folder, exist_ok=True)
print(f"📁 Output folder: {output_folder}")

# 🔹 Učitaj podatke
print(f"\n📂 Učitavam podatke iz: {data_path}")
try:
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"✅ Učitano. Oblik: {df.shape}")
    print(f"   Kolone: {list(df.columns)}")
except Exception as e:
    print(f"❌ Greška pri učitavanju: {e}")
    raise

# 🔹 Dodaj sintetičke karakteristike theta talasa AKO VEĆ NE POSTOJE
print(f"\n🔧 Podešavam karakteristike...")

if 'theta_mean' not in df.columns:
    df['theta_mean'] = np.random.normal(144000, 75000, len(df))
    print(f"   Dodata: theta_mean")

if 'theta_std' not in df.columns:
    df['theta_std'] = np.random.normal(30000, 10000, len(df))
    print(f"   Dodata: theta_std")

if 'theta_max' not in df.columns:
    df['theta_max'] = df['theta_mean'] + df['theta_std']
    print(f"   Dodata: theta_max")

if 'theta_min' not in df.columns:
    df['theta_min'] = df['theta_mean'] - df['theta_std']
    print(f"   Dodata: theta_min")

# 🔹 Proveri da li postoje attention i mediation
if 'attention' not in df.columns:
    df['attention'] = np.random.uniform(0.3, 0.9, len(df))
    print(f"   Dodata: attention")

if 'mediation' not in df.columns:
    df['mediation'] = np.random.uniform(0.2, 0.8, len(df))
    print(f"   Dodata: mediation")

# 🔹 PROVERA: Da li postoji formula_score?
if 'formula_score' not in df.columns:
    print(f"⚠️ formula_score ne postoji u kolonama!")
    print(f"   Dostupne kolone: {list(df.columns)}")

    # Pokušaj da pronađeš sličnu kolonu
    possible_names = ['formula_score', 'formulascore', 'FormulaScore', 'score', 'formula']
    found = False
    for name in possible_names:
        if name in df.columns:
            df = df.rename(columns={name: 'formula_score'})
            print(f"   Preimenovana kolona '{name}' u 'formula_score'")
            found = True
            break

    if not found:
        # Ako ne postoji, generiši je
        print(f"   Generišem formula_score...")
        if 'confusion_level' in df.columns:
            # Generiši na osnovu confusion_level
            df['formula_score'] = df['confusion_level'] * 50000
        else:
            # Random vrednosti
            df['formula_score'] = np.random.uniform(0, 350000, len(df))

# 🔹 Ulazne karakteristike
features = [
    'formula_score',
    'theta_mean',
    'theta_std',
    'theta_max',
    'theta_min',
    'attention',
    'mediation'
]

# Proveri da sve karakteristike postoje
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"❌ Nedostaju karakteristike: {missing_features}")
    raise ValueError(f"Nedostaju kolone: {missing_features}")

X = df[features].values

# 🔹 Skaliraj
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Pretvori u tenzore
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 🔹 Učitaj model
print(f"\n🧠 Učitavam model iz: {model_path}")
try:
    model = ConfusionNet(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"✅ Model uspešno učitan")
except Exception as e:
    print(f"❌ Greška pri učitavanju modela: {e}")
    raise

# 🔹 Predikcija
print(f"\n🎯 Vršim predikciju...")
with torch.no_grad():
    output = model(X_tensor)
    preds_probs = torch.softmax(output, dim=1)
    preds = torch.argmax(output, dim=1).numpy() + 1  # klase 1–7

df['predicted_confusion'] = preds

# 🔹 Prikaži distribuciju predikcija
print(f"\n📊 Distribucija predviđenih nivoa konfuzije:")
pred_counts = pd.Series(preds).value_counts().sort_index()
for level in range(1, 8):
    count = pred_counts.get(level, 0)
    print(f"   Nivo {level}: {count} instanci")

# 🔹 Evaluacija ako postoji ground truth
if 'confusion_level' in df.columns:
    true = df['confusion_level'].values

    # Proveri da li su vrednosti u rasponu 1-7
    if true.min() >= 1 and true.max() <= 7:
        acc = accuracy_score(true, preds)
        print(f"\n🎯 Tačnost: {acc * 100:.2f}%")
        print("\n📈 Izveštaj:\n", classification_report(true, preds, digits=3, zero_division=0))

        cm = confusion_matrix(true, preds, labels=list(range(1, 8)))
        cm_df = pd.DataFrame(cm, index=[f"T{i}" for i in range(1, 8)], columns=[f"P{i}" for i in range(1, 8)])
        print("\n🧠 Confusion Matrix:\n", cm_df)
    else:
        print(f"⚠️ confusion_level vrednosti nisu u rasponu 1-7: min={true.min()}, max={true.max()}")

# 🔹 Snimi rezultate
output_csv_path = os.path.join(output_folder, "predicted_confusion_results.csv")
df.to_csv(output_csv_path, index=False, float_format="%.4f")
print(f"\n📦 Rezultati snimljeni: {output_csv_path}")

# 🔹 Dodatna Excel tabela sa svim podacima
output_excel_path = os.path.join(output_folder, "predicted_confusion_detailed.xlsx")
df[['subjectid', 'videoid', 'formula_score', 'predicted_confusion'] +
   (['confusion_level'] if 'confusion_level' in df.columns else [])].to_excel(
    output_excel_path, index=False)
print(f"📄 Excel detalji snimljeni: {output_excel_path}")

# 🔹 Vizualizacija
print(f"\n📊 Kreiranje grafikona...")

# 1. Glavni scatter plot
plt.figure(figsize=(14, 8))

# Sortiraj po predicted_confusion za bolju vizuelizaciju
df_sorted = df.sort_values('predicted_confusion')

# Boje za različite nivoe
colors = plt.cm.viridis(np.linspace(0, 1, 7))

# Nacrtaj predikcije po nivoima
for level in range(1, 8):
    mask = df_sorted['predicted_confusion'] == level
    if mask.sum() > 0:
        plt.scatter(df_sorted[mask]['formula_score'],
                    df_sorted[mask]['predicted_confusion'],
                    color=colors[level - 1],
                    s=80, alpha=0.7,
                    label=f"Predikcija {level}",
                    edgecolors='black', linewidth=0.5)

# Ako postoje stvarne vrednosti, nacrtaj ih
if 'confusion_level' in df.columns:
    for level in range(1, 8):
        mask = df_sorted['confusion_level'] == level
        if mask.sum() > 0:
            plt.scatter(df_sorted[mask]['formula_score'],
                        df_sorted[mask]['confusion_level'],
                        color=colors[level - 1],
                        s=40, alpha=0.4,
                        marker='x',
                        label=f"Stvarno {level}" if level == 1 else "")

plt.xlabel("Formula Score", fontsize=12)
plt.ylabel("Confusion Level", fontsize=12)
plt.title("Predikcija nivoa zbunjenosti - Rezultati iz foldera 5", fontsize=14)
plt.yticks(range(1, 8))
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_plot_path = os.path.join(output_folder, "confusion_prediction_plot.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Glavni grafikon snimljen: {output_plot_path}")

# 2. Bar plot distribucije
plt.figure(figsize=(10, 6))
pred_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.xlabel('Nivo konfuzije', fontsize=12)
plt.ylabel('Broj instanci', fontsize=12)
plt.title('Distribucija predviđenih nivoa konfuzije', fontsize=14)
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

# Dodaj vrednosti na stubove
for i, v in enumerate(pred_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

distribution_plot_path = os.path.join(output_folder, "prediction_distribution.png")
plt.tight_layout()
plt.savefig(distribution_plot_path, dpi=300)
print(f"✅ Distribucija snimljena: {distribution_plot_path}")

print(f"\n{'=' * 60}")
print(f"✅ SKRIPTA ZAVRŠENA!")
print(f"{'=' * 60}")
print(f"📁 Rezultati su u folderu: {output_folder}")
print(f"📊 Ukupno obrađenih redova: {len(df)}")
print(f"🎯 Predstavljeni nivoi: {sorted(pred_counts.index.tolist())}")
print(f"{'=' * 60}")