"""
Téléchargement des données du projet (tomate uniquement).

- PlantDoc  : version classification (feuilles recadrées), dépôt officiel des auteurs.
- PlantVillage : version "color", uniquement les dossiers tomate (sparse checkout)
  pour éviter de télécharger plusieurs Go inutiles.

Usage :
    python scripts/download_data.py            # télécharge dans ./data/raw
    python scripts/download_data.py --out /content/data/raw

Sources (licence CC BY 4.0, à citer dans le rapport) :
    https://github.com/pratikkayal/PlantDoc-Dataset
    https://github.com/spMohanty/PlantVillage-Dataset
"""
import argparse
import subprocess
from pathlib import Path

PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"

# Dossiers tomate de PlantVillage utilisés (classes communes avec PlantDoc).
# Target_Spot (absente de PlantDoc) et Spider_mites (2 images dans PlantDoc) sont exclues.
PV_TOMATO_DIRS = [
    "raw/color/Tomato___Bacterial_spot",
    "raw/color/Tomato___Early_blight",
    "raw/color/Tomato___Late_blight",
    "raw/color/Tomato___Leaf_Mold",
    "raw/color/Tomato___Septoria_leaf_spot",
    "raw/color/Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "raw/color/Tomato___Tomato_mosaic_virus",
    "raw/color/Tomato___healthy",
]


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def download_plantdoc(out: Path):
    dest = out / "PlantDoc-Dataset"
    if dest.exists():
        print(f"PlantDoc déjà présent : {dest}")
        return
    run(["git", "clone", "--depth", "1", PLANTDOC_URL, str(dest)])


def download_plantvillage(out: Path):
    dest = out / "PlantVillage-Dataset"
    if not dest.exists():
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             PLANTVILLAGE_URL, str(dest)])
    run(["git", "sparse-checkout", "set", "--no-cone", *[d + "/" for d in PV_TOMATO_DIRS]], cwd=dest)
    run(["git", "checkout"], cwd=dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="dossier de destination")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    download_plantdoc(out)
    download_plantvillage(out)

    # Petit récapitulatif
    for d in PV_TOMATO_DIRS:
        p = out / "PlantVillage-Dataset" / d
        n = len(list(p.glob("*"))) if p.exists() else 0
        print(f"PlantVillage {p.name:45s} {n:5d} images")
    print("Téléchargement terminé.")


if __name__ == "__main__":
    main()
