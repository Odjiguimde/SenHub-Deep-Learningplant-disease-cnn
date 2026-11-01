"""
Construit le fichier data/manifest.csv : une ligne par image, avec
    path, source (pv / pd), label (nom de classe unifié), group, split

Choix importants (à expliquer dans le rapport) :
1. Classes : seules les 8 classes tomate présentes dans les DEUX datasets sont gardées,
   avec une table de correspondance explicite (CLASS_MAP ci-dessous).
2. Fuite de données PlantVillage : plusieurs photos d'une même feuille existent.
   Le suffixe du nom de fichier (ex. "GH_HL Leaf 259.1", "GH_HL Leaf 259.2") semble
   identifier la feuille ; on groupe par ce suffixe pour qu'une même feuille ne soit
   jamais à la fois en train et en test (StratifiedGroupKFold).
3. Doublons PlantDoc : PlantDoc vient d'images récupérées sur le web, des doublons exacts
   existent. On les supprime par hash MD5, et on retire du train toute image présente en test.
4. Test PlantDoc : le test officiel ne contient qu'environ 70 images tomate (6 à 11 par classe),
   trop peu pour un F1 par classe stable. Par défaut on refait un découpage stratifié
   (25 % en test). L'option --pd-test-mode official garde le découpage des auteurs.

Usage :
    python scripts/prepare_data.py --raw data/raw --out data/manifest.csv
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# --- Table de correspondance des classes -------------------------------------------
# label unifié : (dossier PlantVillage, dossier PlantDoc)
CLASS_MAP = {
    "bacterial_spot":         ("Tomato___Bacterial_spot",                "Tomato leaf bacterial spot"),
    "early_blight":           ("Tomato___Early_blight",                  "Tomato Early blight leaf"),
    "late_blight":            ("Tomato___Late_blight",                   "Tomato leaf late blight"),
    "leaf_mold":              ("Tomato___Leaf_Mold",                     "Tomato mold leaf"),
    "septoria_leaf_spot":     ("Tomato___Septoria_leaf_spot",            "Tomato Septoria leaf spot"),
    "yellow_leaf_curl_virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    "mosaic_virus":           ("Tomato___Tomato_mosaic_virus",           "Tomato leaf mosaic virus"),
    "healthy":                ("Tomato___healthy",                       "Tomato leaf"),
}
CLASSES = list(CLASS_MAP.keys())
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder} (avez-vous lancé download_data.py ?)")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def pv_group(path: Path, label: str) -> str:
    """'xxxx___GH_HL Leaf 259.1.JPG' -> 'healthy|GH_HL Leaf 259'"""
    stem = path.stem.split("___")[-1]
    stem = re.sub(r"\.\d+$", "", stem)
    return f"{label}|{stem}"


def build_plantvillage(raw: Path, seed: int, max_per_class: int | None) -> pd.DataFrame:
    rows = []
    for label, (pv_dir, _) in CLASS_MAP.items():
        for p in list_images(raw / "PlantVillage-Dataset" / "raw" / "color" / pv_dir):
            rows.append({"path": str(p), "source": "pv", "label": label, "group": pv_group(p, label)})
    df = pd.DataFrame(rows)

    if max_per_class:  # sous-échantillonnage par groupes (utile sur Colab gratuit)
        keep = []
        for label, d in df.groupby("label"):
            groups = d["group"].drop_duplicates().sample(frac=1, random_state=seed)
            sel = d[d["group"].isin(groups)].copy()
            sel["order"] = sel["group"].map({g: i for i, g in enumerate(groups)})
            keep.append(sel.sort_values("order").head(max_per_class).drop(columns="order"))
        df = pd.concat(keep, ignore_index=True)

    # ~15 % test, ~15 % val, ~70 % train, sans qu'un groupe soit coupé en deux
    df["split"] = "pv_train"
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], df["group"]))
    df.loc[df.index[test_idx], "split"] = "pv_test"
    rest = df[df["split"] == "pv_train"]
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(rest, rest["label"], rest["group"]))
    df.loc[rest.index[val_idx], "split"] = "pv_val"
    return df


def build_plantdoc(raw: Path, seed: int, test_mode: str, test_frac: float) -> pd.DataFrame:
    rows = []
    for official in ("train", "test"):
        for label, (_, pd_dir) in CLASS_MAP.items():
            for p in list_images(raw / "PlantDoc-Dataset" / official / pd_dir):
                rows.append({"path": str(p), "source": "pd", "label": label, "official": official})
    df = pd.DataFrame(rows)

    n0 = len(df)
    df = df[df["path"].map(lambda s: is_valid_image(Path(s)))].copy()
    print(f"PlantDoc : {n0 - len(df)} image(s) illisible(s) retirée(s)")

    df["md5"] = df["path"].map(lambda s: md5(Path(s)))
    df["group"] = df["md5"]

    if test_mode == "official":
        test_md5 = set(df.loc[df["official"] == "test", "md5"])
        leak = (df["official"] == "train") & df["md5"].isin(test_md5)
        print(f"PlantDoc : {leak.sum()} image(s) du train identiques à une image du test, retirée(s)")
        df = df[~leak].drop_duplicates("md5").copy()
        df["split"] = df["official"].map({"train": "pd_train", "test": "pd_test"})
    else:
        n1 = len(df)
        # une image présente sous deux labels différents est ambiguë : on la retire
        ambiguous = df.groupby("md5")["label"].nunique()
        df = df[df["md5"].isin(ambiguous[ambiguous == 1].index)]
        df = df.drop_duplicates("md5").copy()
        print(f"PlantDoc : {n1 - len(df)} doublon(s) exact(s) retiré(s)")
        tr, te = train_test_split(df.index, test_size=test_frac, stratify=df["label"], random_state=seed)
        df.loc[tr, "split"] = "pd_train"
        df.loc[te, "split"] = "pd_test"

    tr = df[df["split"] == "pd_train"]
    _, val_idx = train_test_split(tr.index, test_size=0.2, stratify=tr["label"], random_state=seed)
    df.loc[val_idx, "split"] = "pd_val"
    return df.drop(columns=["official", "md5"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pd-test-mode", choices=["resplit", "official"], default="resplit")
    parser.add_argument("--pd-test-frac", type=float, default=0.25)
    parser.add_argument("--pv-max-per-class", type=int, default=None,
                        help="limite d'images PlantVillage par classe (ex. 1000 sur Colab gratuit)")
    args = parser.parse_args()

    raw = Path(args.raw)
    pv = build_plantvillage(raw, args.seed, args.pv_max_per_class)
    pdoc = build_plantdoc(raw, args.seed, args.pd_test_mode, args.pd_test_frac)
    manifest = pd.concat([pv, pdoc], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)

    print(f"\nManifest écrit : {out} ({len(manifest)} images)\n")
    table = manifest.pivot_table(index="label", columns="split", values="path",
                                 aggfunc="count", fill_value=0)
    order = [c for c in ["pv_train", "pv_val", "pv_test", "pd_train", "pd_val", "pd_test"] if c in table]
    print(table[order].reindex(CLASSES).to_string())

    # Contrôle anti-fuite : aucun groupe PlantVillage ne doit apparaître dans deux splits
    shared = manifest[manifest.source == "pv"].groupby("group")["split"].nunique()
    assert (shared == 1).all(), "Fuite : un groupe PlantVillage est présent dans plusieurs splits"
    print("\nContrôle anti-fuite PlantVillage : OK")


if __name__ == "__main__":
    main()
