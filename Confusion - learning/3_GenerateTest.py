import numpy as np
import pandas as pd
import os

# 🔹 Definiši putanju do glavnog foldera
base_path = "C:/Programi_Python/Bad learning/Confusion - bad learning"

# 🔹 Sada generišemo fajl direktno u glavnom folderu (gde su folderi 1, 2, 4, 5, 6)
output_path = os.path.join(base_path, "test_eeg_data.csv")

print(f"📁 Izlazni fajl će biti kreiran na: {output_path}")

# 🔹 Pokušaj da nađeš originalni EEG_data.csv u glavnom folderu
possible_paths = [
    os.path.join(base_path, "EEG_data.csv"),  # Glavni folder gde su 1, 2, 4, 5, 6
    "EEG_data.csv",  # Trenutni folder
    "../EEG_data.csv",  # Parent folder
    "./EEG_data.csv",  # Trenutni folder
    os.path.join(os.path.dirname(__file__), "EEG_data.csv"),  # Folder gde je script
    "C:/Programi_Python/PhD_EEG/EEG_data.csv",  # Originalna putanja
]

eeg_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        eeg_data_path = path
        print(f"✅ Pronađen EEG_data.csv na: {path}")
        break

if eeg_data_path is None:
    # Proveri da li postoji u folderima 1, 2, 4, 5, 6
    for folder in ["1", "2", "4", "5", "6"]:
        folder_path = os.path.join(base_path, folder, "EEG_data.csv")
        if os.path.exists(folder_path):
            eeg_data_path = folder_path
            print(f"✅ Pronađen EEG_data.csv u folderu '{folder}': {folder_path}")
            break

    if eeg_data_path is None:
        print("❌ EEG_data.csv nije pronađen na uobičajenim lokacijama.")
        print(f"🔍 Tražim CSV fajlove u: {base_path}")
        current_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

        if current_files:
            print(f"\n📋 Dostupni CSV fajlovi u glavnom folderu:")
            for i, f in enumerate(current_files, 1):
                print(f"   {i}. {f}")

            try:
                choice = int(input(f"\n🎯 Unesite broj fajla koji želite da koristite (1-{len(current_files)}): "))
                if 1 <= choice <= len(current_files):
                    eeg_data_path = os.path.join(base_path, current_files[choice - 1])
                    print(f"✅ Odabran fajl: {eeg_data_path}")
                else:
                    print("❌ Nevažeći izbor. Prekidam.")
                    exit(1)
            except:
                print("❌ Nevažeći unos. Prekidam.")
                exit(1)
        else:
            raise FileNotFoundError(f"❌ Nema CSV fajlova u {base_path}. Postavite EEG_data.csv.")

# 🔹 Učitaj originalni EEG fajl
try:
    original_df = pd.read_csv(eeg_data_path)
    print(f"✅ Fajl uspešno učitan. Oblik: {original_df.shape}")
except Exception as e:
    print(f"❌ Greška pri učitavanju {eeg_data_path}: {e}")
    exit(1)

# 🔹 Kolone koje ćemo imitirati
target_columns = [
    "Attention", "Mediation", "Raw", "Delta", "Theta",
    "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"
]

# 🔹 Proveri da sve kolone postoje
missing = [col for col in target_columns if col not in original_df.columns]
if missing:
    print(f"⚠️ Nedostaju neke kolone u {eeg_data_path}: {missing}")
    print(f"📋 Dostupne kolone: {list(original_df.columns)}")

    # Proba da pronađe slične nazive kolona (case insensitive)
    available_cols = list(original_df.columns)
    found_cols = []
    new_targets = []

    for target in target_columns:
        # Proveri različite varijacije
        variations = [
            target,
            target.lower(),
            target.upper(),
            target.capitalize(),
            target.replace(" ", ""),
            target.replace("-", ""),
            target.replace("_", "")
        ]

        found = False
        for var in variations:
            if var in available_cols:
                found_cols.append(var)
                new_targets.append(target)
                found = True
                break

        if not found:
            print(f"   ❌ {target} nije pronađena ni u kojem obliku")

    if len(found_cols) == len(target_columns):
        print(f"✅ Pronađene sve kolone u drugačijem formatu")
        # Mapiranje starih naziva na nove
        col_mapping = dict(zip(target_columns, found_cols))
        original_df = original_df.rename(columns=col_mapping)
    else:
        raise ValueError(f"❌ Nedostaju esencijalne kolone. Dostupne: {available_cols}")

# 🔹 Izvuci distribucije iz originala
stats = {}
for col in target_columns:
    series = original_df[col].dropna()
    if len(series) == 0:
        print(f"⚠️ Kolona {col} je prazna! Koristim podrazumevane vrednosti.")
        stats[col] = {
            "mean": 50.0,
            "std": 10.0,
            "min": 0.0,
            "max": 100.0,
            "values": np.array([50.0])
        }
    else:
        stats[col] = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "values": series.values
        }

# 🔹 Parametri
num_levels = 7
graphs_per_level = 100
samples = 140
rows = []


# 🔹 Funkcija za generisanje jednog reda
def generate_row(level, subject_id, video_id):
    values = {}

    for col in target_columns:
        if col == "Theta":
            # Theta direktno iz distribucije bez skaliranja
            val = np.random.choice(stats[col]["values"])
        elif col in ["Attention", "Mediation"]:
            # Attention & Mediation iz distribucije + jitter ±3%
            base = np.random.choice(stats[col]["values"])
            jitter = np.random.normal(loc=1.0, scale=0.03)
            val = np.clip(base * jitter, stats[col]["min"], stats[col]["max"])
        else:
            # Ostalo iz normalne distribucije sa preciznom std
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            val = np.random.normal(loc=mean, scale=std)
            val = np.clip(val, stats[col]["min"], stats[col]["max"])
        values[col] = val

    predefined = level
    user_defined = level

    return [
        subject_id, video_id,
        values["Attention"], values["Mediation"], values["Raw"], values["Delta"], values["Theta"],
        values["Alpha1"], values["Alpha2"], values["Beta1"], values["Beta2"], values["Gamma1"], values["Gamma2"],
        predefined, user_defined
    ]


# 🔹 Generiši podatke
print("\n🔨 Generišem podatke...")
for level in range(1, num_levels + 1):
    print(f"   Nivo {level}/{num_levels}...")
    for i in range(graphs_per_level):
        subject_id = f"synthetic_{level}_{i + 1}"
        video_id = f"confusion_{level}"
        for t in range(samples):
            row = generate_row(level, subject_id, video_id)
            rows.append(row)

# 🔹 Zaglavlje
columns = [
    "SubjectID", "VideoID", "Attention", "Mediation", "Raw", "Delta", "Theta",
    "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2",
    "predefinedlabel", "user-definedlabeln"
]

# 🔹 Snimi CSV direktno u glavni folder (pored foldera 1, 2, 4, 5, 6)
df = pd.DataFrame(rows, columns=columns)
df.to_csv(output_path, index=False, float_format="%.6f")
print(f"\n✅ test_eeg_data.csv generisan na lokaciji:")
print(f"   {output_path}")
print(f"   Broj redova: {len(df):,}")
print(f"   Broj kolona: {len(df.columns)}")
print("   Theta, attention i mediation sada statistički verno prate original.")