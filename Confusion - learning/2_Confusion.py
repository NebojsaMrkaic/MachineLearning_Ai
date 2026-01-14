import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 🔹 Proveri da li postoje fajlovi u folderu "1"
folder_1 = "1"
csv_files = [f for f in os.listdir(folder_1) if f.endswith('.csv')]

if not csv_files:
    print(f"❌ Nema CSV fajlova u folderu '{folder_1}'. Proverite da li folder i fajlovi postoje.")
    exit(1)

# Pronađi fajlove koji sadrže 'attention' ili 'mediation' u imenu
attention_files = [f for f in csv_files if 'attention' in f.lower() or 'mediation' in f.lower()]

if not attention_files:
    print(f"❌ Nije pronađen fajl sa 'attention' ili 'mediation' u imenu u folderu '{folder_1}'.")
    print(f"   Dostupni fajlovi: {csv_files}")
    exit(1)

# Uzmi prvi pronađeni fajl
csv_filename = attention_files[0]
csv_path = os.path.join(folder_1, csv_filename)

print(f"📂 Učitavam fajl: {csv_path}")

# 🔹 Kreiraj folder "2" za sve izlaze
os.makedirs("2", exist_ok=True)

# 🔹 Učitaj attention/mediation podatke
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Fajl uspešno učitan. Oblik podataka: {df.shape}")
except Exception as e:
    print(f"❌ Greška pri učitavanju fajla: {e}")
    exit(1)

# 🔹 Preprocesiranje kolona
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("%", "")
df.rename(columns={'studentid': 'student', 'videoid': 'video'}, inplace=True)

print(f"📊 Kolone u dataframe-u: {list(df.columns)}")

# 🔹 Prosečan attention i mediation po studentu
avg_df = df.groupby('student')[['attention', 'mediation']].mean().reset_index()

# 🔹 Bar grafikon prosečnog attention po studentu
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(avg_df['student'], avg_df['attention'], color='mediumseagreen')
ax.set_xlabel("Student ID")
ax.set_ylabel("Prosečan Attention %")
ax.set_title("Prosečna pažnja po studentu")
fig.subplots_adjust(bottom=0.15)
plt.savefig("2/avg_attention_per_student.png", dpi=300)
plt.close()
print("✅ Grafikon prosečne pažnje je snimljen kao '2/avg_attention_per_student.png'")

# 🔹 Bar grafikon prosečnog mediation po studentu
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(avg_df['student'], avg_df['mediation'], color='cornflowerblue')
ax.set_xlabel("Student ID")
ax.set_ylabel("Prosečan Mediation %")
ax.set_title("Prosečna mediacija po studentu")
fig.subplots_adjust(bottom=0.15)
plt.savefig("2/avg_mediation_per_student.png", dpi=300)
plt.close()
print("✅ Grafikon prosečne meditacije je snimljen kao '2/avg_mediation_per_student.png'")

# 🔹 Scatter grafikon attention vs mediation za sve snimke
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['attention'], df['mediation'], c=df['student'], cmap='tab10', s=80, edgecolors='k')
ax.set_xlabel("Attention %")
ax.set_ylabel("Mediation %")
ax.set_title("Attention vs Mediation po snimku")
plt.colorbar(sc, label="Student ID")
fig.subplots_adjust(bottom=0.15)
plt.savefig("2/attention_vs_mediation_scatter.png", dpi=300)
plt.close()
print("✅ Scatter grafikon pažnje vs meditacije je snimljen kao '2/attention_vs_mediation_scatter.png'")

print(f"\n🎯 Svi grafikoni su sačuvani u folderu '2/'")