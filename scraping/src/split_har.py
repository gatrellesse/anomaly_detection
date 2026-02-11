import json
import os
import copy
import sys

def split_har(input_har, output_dir):
    # Charger le fichier HAR
    with open(input_har, "r", encoding="utf-8") as f:
        har = json.load(f)

    log = har.get("log", {})
    entries = log.get("entries", [])

    if not entries:
        print("Aucune entry trouvée dans le fichier HAR.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(entries):
        new_har = {
            "log": copy.deepcopy(log)
        }

        # Garder une seule entry
        new_har["log"]["entries"] = [entry]

        output_path = os.path.join(output_dir, f"entry_{i+1:04d}.har")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_har, f, indent=2, ensure_ascii=False)

    print(f"{len(entries)} fichiers HAR générés dans '{output_dir}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_har.py input.har output_dir")
        sys.exit(1)

    split_har(sys.argv[1], sys.argv[2])

#python split_har.py mon_fichier.har har_split
