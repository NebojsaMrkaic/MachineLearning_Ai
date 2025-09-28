import os
import shutil

# Putanje do foldera
labels_dir = "C:/Users/Korisnik/Desktop/Python/1/brain-tumor/valid/labels"
images_dir = "C:/Users/Korisnik/Desktop/Python/1/brain-tumor/valid/images"
output_dir = "C:/Users/Korisnik/Desktop/Python/1/brain-tumor/valid_classified"

# Kreiraj positive/negative foldere
os.makedirs(f"{output_dir}/positive", exist_ok=True)
os.makedirs(f"{output_dir}/negative", exist_ok=True)

# Prolazak kroz sve .txt fajlove
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(images_dir, image_name)

    # Proveri da li slika postoji
    if not os.path.exists(image_path):
        print(f"⚠️ Slika ne postoji: {image_name}")
        continue

    # Pročitaj prvu liniju iz label fajla
    with open(label_path, "r") as f:
        first_line = f.readline().strip()

    if not first_line:
        print(f"⚠️ Prazan label fajl: {label_file}")
        continue

    # Uhvati klasu (0 ili 1)
    class_id = first_line.split()[0]

    if class_id == "0":
        shutil.copy(image_path, f"{output_dir}/negative/{image_name}")
    elif class_id == "1":
        shutil.copy(image_path, f"{output_dir}/positive/{image_name}")
    else:
        print(f"⚠️ Nepoznata klasa u {label_file}: {class_id}")

print("✅ Slike su razvrstane po klasama u positive/negative foldere.")
