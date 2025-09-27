# 1. Uvoz biblioteka
import tensorflow as tf  # Glavna biblioteka za pravljenje i treniranje ML modela
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Alat za učitavanje i pripremu slika
import matplotlib.pyplot as plt  # Za crtanje grafika (npr. tačnost kroz epohe)

# 2. Putanje do foldera sa slikama
train_dir = "C:\\Users\\Korisnik\\Desktop\\Python\\1\\brain-tumor\\train_classified"  # Folder sa treniranjem
val_dir = "C:\\Users\\Korisnik\\Desktop\\Python\\1\\brain-tumor\\valid_classified"    # Folder sa validacijom

# 3. Priprema slika za treniranje (augmentacija + normalizacija)
train_gen = ImageDataGenerator(
    rescale=1./255,             # Pikseli idu od 0 do 1 umesto 0 do 255
    rotation_range=10,          # Rotira slike do 10 stepeni
    zoom_range=0.1,             # Zumira slike do 10%
    horizontal_flip=True        # Prevrće slike horizontalno
)

# 4. Priprema slika za validaciju (samo normalizacija)
val_gen = ImageDataGenerator(rescale=1./255)

# 5. Učitavanje slika iz foldera
train_data = train_gen.flow_from_directory(
    train_dir,                  # Gde su slike
    target_size=(256, 256),     # Menja veličinu slika na 256x256
    batch_size=32,              # Učitava po 32 slike odjednom
    class_mode='binary'         # Dve klase: tumor / bez tumora
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# 6. Pravljenje CNN modela
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),  # Prvi sloj: 32 filtera
    tf.keras.layers.MaxPooling2D(2,2),                                               # Smanjuje sliku na pola

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                            # Drugi sloj: 64 filtera
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),                           # Treći sloj: 128 filtera
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),                                                      # Pretvara sliku u niz brojeva
    tf.keras.layers.Dense(128, activation='relu'),                                  # Dodaje neuronski sloj
    tf.keras.layers.Dropout(0.3),                                                   # Gasi 30% neurona nasumično
    tf.keras.layers.Dense(1, activation='sigmoid')                                  # Završni sloj: 0 ili 1
])

# 7. Kompajliranje modela
model.compile(
    optimizer='adam',                 # Pametan algoritam za učenje
    loss='binary_crossentropy',       # Greška za dve klase
    metrics=['accuracy']              # Prati tačnost
)

# 8. Treniranje modela
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10                         # Treniraj 10 puta kroz sve slike
)

# 9. Evaluacija modela
loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc:.2f}")  # Prikazuje tačnost na slikama koje model nije video

# 10. Crtanje grafika tačnosti
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Tačnost kroz epohe")
plt.xlabel("Epohe")
plt.ylabel("Tačnost")
plt.show()

# 11. Čuvanje modela
model.save("tumor_model.h5")
print("✅ Model je sačuvan kao tumor_model.h5")
