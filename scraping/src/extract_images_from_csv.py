import pandas as pd
import os
import requests

csv_folder = "/home/vince/ENSTA/4A/projet/scraping/csv/"

# Lister tous les fichiers CSV du dossier
csv_files = [
    os.path.join(csv_folder, f)
    for f in os.listdir(csv_folder)
    if f.lower().endswith(".csv")
]

# Lire et concaténer tous les CSV
df = pd.concat(
    (pd.read_csv(csv_file) for csv_file in csv_files),
    ignore_index=True
)


# Nom de la colonne contenant les URL
url_column = "images.urls"

# Dossier où les images seront téléchargées
output_folder = "/home/vince/ENSTA/4A/projet/scraping/images_extraites_lbc/vetements_troues"

print(f"Le .csv contient {len(df)} lignes.")

# Compteur pour nommer les fichiers
image_counter = 1

# Parcourir toutes les cellules
for k, cell in enumerate(df[url_column]):
    print(f"Extraction des images de la ligne n°{k} du .csv :")
    if pd.isna(cell):
        continue  # ignorer les cellules vides
    urls = cell.split("|")  # séparer les URL
    for url in urls:
        url = url.strip()  # enlever les espaces
        if not url:
            continue
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Déterminer l'extension du fichier
            file_ext = os.path.splitext(url)[1]
            if not file_ext or len(file_ext) > 5:  # si extension bizarre ou trop longue
                file_ext = ".jpg"
            file_name = f"image_{image_counter}{file_ext}"
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Téléchargé {file_name}")
            image_counter += 1
        except Exception as e:
            print(f"Erreur pour l'URL {url}: {e}")