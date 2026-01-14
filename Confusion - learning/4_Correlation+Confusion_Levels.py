import pandas as pd
import plotly.express as px
import os
import numpy as np

# 🔹 Kreiraj folder '4' ako ne postoji
os.makedirs("4", exist_ok=True)


# 🔹 Funkcija za čišćenje kolona
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")
    return df


# 🔹 Funkcija za korelaciju
def plot_correlation(df, label):
    corr = df[['theta', 'attention', 'mediation']].dropna().corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        title=f"Korelacija — {label}"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    fig.write_image(f"4/correlation_{label}.png")
    print(f"✅ Korelacija snimljena: 4/correlation_{label}.png")


# 🔹 Funkcija za formulu, scatter plot i Excel sa confusion_level
def plot_formula(df, label):
    grouped = df.groupby(['subjectid', 'videoid'])
    excel_rows = []

    for (subject, video), group in grouped:
        theta = group['theta'].mean()
        attention = group['attention'].mean()
        mediation = group['mediation'].mean()

        # 🔹 DODATA PROVERA: Izbegni deljenje sa 0 i ekstremne vrednosti
        if mediation == 0 or pd.isna(mediation) or pd.isna(attention) or pd.isna(theta):
            continue

        # 🔹 KLJUČNA PROMENA: Normalizuj attention i mediation pre računanja formule
        # Originalna formula: (attention / mediation) * theta može dati ogromne brojeve
        # Ako je attention 80, mediation 20, theta 60 → (80/20)*60 = 4*60 = 240

        # PRVA OPREZIJA: Ograniči odnos attention/mediation
        attention_mediation_ratio = attention / mediation

        # Ako je ratio previsok (npr. >5), skaliraj ga
        if attention_mediation_ratio > 5:
            attention_mediation_ratio = 5  # Maksimalni ratio

        # DRUGA OPREZIJA: Skaliraj theta na normalan raspon
        # Ako je theta previsoka (npr. >100), skaliraj je
        theta_scaled = min(theta, 100)

        # NOVA FORMULA sa skaliranim vrednostima
        score = attention_mediation_ratio * theta_scaled

        # TREĆA OPREZIJA: Dodaj log transformaciju ako je potrebno
        if score > 500:  # Ako je još uvek preveliko
            score = 100 + np.log10(score) * 100  # Log transformacija

        excel_rows.append({
            'subjectid': subject,
            'videoid': video,
            'attention': attention,
            'mediation': mediation,
            'theta': theta,
            'formula_score': score
        })

    if not excel_rows:
        print(f"⚠️ Nema validnih formula za {label}")
        return

    # 🔹 Snimi Excel tabelu
    df_excel = pd.DataFrame(excel_rows)

    # 🔹 OČUVANJE ORIGINALNE SKALE 0-350k ZA Y OSU
    # Normalizuj formula_score da bude u rasponu 0-350000
    min_score = df_excel['formula_score'].min()
    max_score = df_excel['formula_score'].max()

    # Skaliraj na 0-350000
    df_excel['formula_score_scaled'] = df_excel['formula_score'].apply(
        lambda x: 350000 * (x - min_score) / (max_score - min_score) if max_score != min_score else 175000
    )

    # 🔹 Dodaj confusion_level skaliran od 1 do 7
    df_excel['confusion_level'] = df_excel['formula_score_scaled'].apply(
        lambda x: round(1 + 6 * (x - df_excel['formula_score_scaled'].min()) /
                        (df_excel['formula_score_scaled'].max() - df_excel['formula_score_scaled'].min()))
    )

    # 🔹 Snimi Excel tabelu sa oba rezultata
    df_excel.to_excel(f"4/formula_values_{label}.xlsx", index=False)
    print(f"📄 Excel snimljen: 4/formula_values_{label}.xlsx")

    # 🔹 Sortiraj za scatter plot - KORISTI SKALIRANU VREDNOST za Y osu
    df_excel_sorted = df_excel.sort_values(by='formula_score_scaled', ascending=False)

    # 🔹 Kreiraj scatter plot sa FIKSNOM Y OSOM (0-350k)
    fig = px.scatter(
        x=df_excel_sorted['subjectid'].astype(str) + "-" + df_excel_sorted['videoid'].astype(str),
        y=df_excel_sorted['formula_score_scaled'],
        color=df_excel_sorted['formula_score_scaled'],
        color_continuous_scale='Viridis',
        title=f"Učenje i razumevanje — {label}",
        labels={'x': 'Student-Video', 'y': 'Formula Score (0-350k)'},
        range_y=[0, 350000]  # 🔹 FIKSNI RASPON Y OSE!
    )

    # 🔹 Dodaj horizontalne linije za reference
    fig.add_hline(y=350000, line_dash="dot", line_color="red",
                  annotation_text="Max: 350k", annotation_position="top left")
    fig.add_hline(y=175000, line_dash="dash", line_color="orange",
                  annotation_text="Sredina: 175k", annotation_position="top left")
    fig.add_hline(y=0, line_dash="solid", line_color="blue",
                  annotation_text="Min: 0", annotation_position="bottom left")

    fig.update_layout(
        xaxis_tickangle=45,
        yaxis=dict(
            title='Formula Score',
            tickmode='linear',
            tick0=0,
            dtick=50000,  # Koraci od 50k
            range=[0, 350000]  # 🔹 FIKSNI RASPON
        ),
        margin=dict(l=40, r=40, t=40, b=80)
    )

    fig.write_image(f"4/formula_{label}.png")
    print(f"✅ Formula scatter snimljen: 4/formula_{label}.png")

    # 🔹 Dodatna analiza
    print(f"\n📊 ANALIZA FORMULE ZA {label}:")
    print(f"   Originalni min score: {min_score:.2f}")
    print(f"   Originalni max score: {max_score:.2f}")
    print(f"   Skalirani min: {df_excel['formula_score_scaled'].min():.2f}")
    print(f"   Skalirani max: {df_excel['formula_score_scaled'].max():.2f}")
    print(f"   Broj tačaka: {len(df_excel)}")


# 🔹 Obradi oba fajla
for file in ["EEG_data.csv", "test_EEG_data.csv"]:
    if os.path.exists(file):
        label = file.replace(".csv", "")
        print(f"\n{'=' * 60}")
        print(f"📂 Obrađujem: {file}")
        print(f"{'=' * 60}")
        df = pd.read_csv(file)
        df = clean_columns(df)
        plot_correlation(df, label)
        plot_formula(df, label)
    else:
        print(f"⚠️ Fajl {file} ne postoji! Preskačem...")