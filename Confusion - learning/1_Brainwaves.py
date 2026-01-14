import pandas as pd
import matplotlib.pyplot as plt
import os

# 🔹 Učitaj EEG podatke
eeg_df = pd.read_csv("EEG_data.csv")
eeg_df.columns = eeg_df.columns.str.strip().str.lower().str.replace(" ", "")

# 🔹 Učitaj demografske podatke
demo_df = pd.read_csv("demographic_info.csv")
demo_df.columns = demo_df.columns.str.strip().str.lower().str.replace(" ", "")

# 🔹 Spoji EEG + demografiju
merged_df = pd.merge(eeg_df, demo_df, on='subjectid', how='left')

# 🔹 Napravi folder "1" za sve izlaze
os.makedirs("1", exist_ok=True)

# 🔹 Jedinstvene kombinacije
subjects = merged_df['subjectid'].unique()
videos = merged_df['videoid'].unique()

# 🔹 Petlja kroz sve subjekte i video snimke
for sid in subjects:
    for vid in videos:
        sample = merged_df[(merged_df['subjectid'] == sid) & (merged_df['videoid'] == vid)]
        if sample.empty:
            continue

        age = sample['age'].iloc[0]
        gender = sample['gender'].iloc[0]
        ethnicity = sample['ethnicity'].iloc[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1.plot(sample['delta'].values, label='Delta (0.5–4 Hz)', alpha=0.6)
        ax1.plot(sample['theta'].values, label='Theta (4–8 Hz)', alpha=0.6)
        ax1.plot(sample['alpha1'].values, label='Alpha (8–13 Hz)', alpha=0.6)
        ax1.plot(sample['beta1'].values, label='Beta (13–30 Hz)', alpha=0.6)
        ax1.set_ylabel("EEG Amplituda")
        ax1.set_title(f"EEG Talasi – Student {sid} ({gender}, {age}, {ethnicity}) – Video {vid}")
        ax1.legend(loc='upper right')
        ax1.grid(True)

        ax2.plot(sample['attention'].values, label='Attention', color='black', linestyle='--', linewidth=1.5)
        ax2.plot(sample['mediation'].values, label='Mediation', color='purple', linestyle=':', linewidth=1.5)
        ax2.set_xlabel("Vremenska tačka (0.5s)")
        ax2.set_ylabel("Kognitivni signal")
        ax2.set_title("Attention i Mediation kroz vreme")
        ax2.legend(loc='upper right')
        ax2.grid(True)

        fig.subplots_adjust(hspace=0.3)
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=6)

        filename = f"1/eeg_dualplot_s{sid}_v{vid}.png"
        try:
            plt.savefig(filename, dpi=300)
            print(f"✅ Snimljen grafikon: {filename}")
        except Exception as e:
            print(f"❌ Greška pri snimanju {filename}: {e}")
        finally:
            plt.close()

print("✅ Svi grafovi su snimljeni u folder '1'")

# 🔹 Grupisanje po studentu i videu
attention_table = merged_df.groupby(['subjectid', 'videoid'])[['attention', 'mediation']].mean().reset_index()
attention_table['attention'] = attention_table['attention'].round(2)
attention_table['mediation'] = attention_table['mediation'].round(2)
attention_table.columns = ['Student ID', 'Video ID', 'Attention %', 'Mediation %']

print("\n📊 Prosečna pažnja i mediation po studentu i videu:\n")
print(attention_table)

# 🔹 Snimi tabelu kao CSV u folder "1"
attention_table.to_csv("1/attention_mediation_summary.csv", index=False)
print("✅ Tabela je snimljena kao '1/attention_mediation_summary.csv'")

# 🔹 Attention barplot
attention_sorted = attention_table.sort_values(by='Attention %', ascending=False).reset_index(drop=True)
x_labels_att = attention_sorted.apply(lambda row: f"{row['Student ID']}-V{row['Video ID']}", axis=1)
x_att = range(len(x_labels_att))

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(x_att, attention_sorted['Attention %'], width=0.6, color='skyblue')
ax.set_xticks(x_att)
ax.set_xticklabels(x_labels_att, rotation=30, ha='right', fontsize=9)
ax.set_ylabel("Attention %")
ax.set_title("Attention po studentu i videu (sortirano)")
fig.subplots_adjust(bottom=0.2)
plt.locator_params(axis='x', nbins=10)
plt.savefig("1/attention_sorted_barplot.png", dpi=300)
plt.close()
print("✅ Bar grafikon za Attention je snimljen kao '1/attention_sorted_barplot.png'")

# 🔹 Mediation barplot
mediation_sorted = attention_table.sort_values(by='Mediation %', ascending=False).reset_index(drop=True)
x_labels_med = mediation_sorted.apply(lambda row: f"{row['Student ID']}-V{row['Video ID']}", axis=1)
x_med = range(len(x_labels_med))

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(x_med, mediation_sorted['Mediation %'], width=0.6, color='orchid')
ax.set_xticks(x_med)
ax.set_xticklabels(x_labels_med, rotation=30, ha='right', fontsize=9)
ax.set_ylabel("Mediation %")
ax.set_title("Mediation po studentu i videu (sortirano)")
fig.subplots_adjust(bottom=0.2)
plt.locator_params(axis='x', nbins=10)
plt.savefig("1/mediation_sorted_barplot.png", dpi=300)
plt.close()
print("✅ Bar grafikon za Mediation je snimljen kao '1/mediation_sorted_barplot.png'")
