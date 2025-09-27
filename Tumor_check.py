import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 1. Učitavanje modela
model = tf.keras.models.load_model("tumor_model.h5")

# 2. Putanja do slike koju želiš da testiraš
img_path = "C:/Users/Korisnik/Desktop/Python/1/datasets/brain-tumor/images/train/00054_145.jpg"  # Promeni ako koristiš drugo ime

# 3. Učitavanje i priprema slike
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 4. Predikcija
prediction = model.predict(img_array)

# 5. Rezultat
if prediction[0][0] > 0.5:
    print("✅ Tumor detektovan")
else:
    print("🧠 Nema tumora")

# 6. Prikaz slike (opciono)
plt.imshow(img)
plt.title("MRI slika za testiranje")
plt.axis("off")
plt.show()
