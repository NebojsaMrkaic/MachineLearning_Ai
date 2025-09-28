import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 🔹 Učitaj podatke
df = pd.read_csv("airlines_flights_data.csv")
df.columns = df.columns.str.strip()  # ukloni razmake

# 🔹 Enkoduj kategorije
for col in ['airline', 'source_city', 'destination_city', 'class']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 🔹 Priprema podataka
X = df[['airline', 'source_city', 'destination_city', 'class', 'duration', 'days_left']]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Definiši TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 🔹 Treniraj model sa prikazom epoha
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# 🔹 Evaluacija
test_loss = model.evaluate(X_test, y_test)
print(f"\n📉 Test Loss (MSE): {test_loss:.2f}")

# 🔹 Predikcije
predictions = model.predict(X_test)

print("\n🔍 Prvih 10 predikcija vs stvarna cena:")
for i in range(10):
    print(f"🔮 Predikcija: {predictions[i][0]:.2f} | 🎯 Stvarna cena: {y_test.iloc[i]:.2f}")

# 🔹 Vizualizacija predikcija vs stvarna cena
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.6, label="Predikcije")

# 🔹 Linija idealne predikcije
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Idealna linija")

plt.xlabel("Stvarna cena")
plt.ylabel("Predikcija")
plt.title("Predikcija cene avionske karte")
plt.legend()
plt.grid(True)

# 🔹 Snimi graf
plt.savefig("predikcija_cena.png", dpi=300)
plt.show()

# 🔹 Vizualizacija loss-a po epohama
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Trening Loss')
plt.plot(history.history['val_loss'], label='Validacioni Loss')
plt.xlabel("Epoha")
plt.ylabel("Loss (MSE)")
plt.title("Loss tokom treniranja")
plt.legend()
plt.grid(True)
plt.savefig("loss_po_epohama.png", dpi=300)
plt.show()
