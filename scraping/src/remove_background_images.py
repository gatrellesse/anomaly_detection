#pip install rembg[gpu]

import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rembg import remove

# --- Configuration ---
PATH_IMAGES = "/home/vince/ENSTA/4A/projet/scraping/datasets/jeans_troues/troues"
PATH_BG_REMOVED = "/home/vince/ENSTA/4A/projet/scraping/datasets/jeans_troues_no_bg/troues"

def process_background_removal():
    # Liste des extensions d'images supportées
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    images_list = [f for f in os.listdir(PATH_IMAGES) if f.lower().endswith(extensions)]

    print(f"Traitement de {len(images_list)} images...")

    for filename in tqdm(images_list):
        input_path = os.path.join(PATH_IMAGES, filename)
        output_path = os.path.join(PATH_BG_REMOVED, filename)

        # 1. Ouvrir l'image
        input_image = Image.open(input_path)

        # 2. Supprimer le background (donne une image avec transparence RGBA)
        output_image = remove(input_image)

        # 3. Créer un fond blanc
        # On crée une nouvelle image RGB remplie de blanc de la même taille
        white_background = Image.new("RGB", output_image.size, (255, 255, 255))

        # 4. Superposer l'image sans fond sur le fond blanc
        # Le masque est l'image elle-même car elle possède un canal alpha
        white_background.paste(output_image, (0, 0), output_image)

        # 5. Sauvegarder le résultat
        # On force la sauvegarde en .jpg ou .png (le fond est maintenant opaque)
        white_background.save(output_path)

if __name__ == "__main__":
    process_background_removal()
    print(f"Terminé ! Les images sont disponibles dans : {PATH_BG_REMOVED}")