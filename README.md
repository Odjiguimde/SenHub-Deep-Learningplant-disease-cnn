# SenHub-Deep-Learningplant-disease-cnn
# Détection des maladies de la tomate par CNN : du laboratoire au champ

Projet de fin de module Deep Learning, cohorte 3 AI Hub.
**Groupe** : *DJIGUIMDE Oumaro Titans*

Une coopérative agricole veut qu'un agriculteur puisse photographier une feuille de tomate avec son
smartphone et obtenir un diagnostic préliminaire. Nous entraînons des CNN conçus from scratch sur des
images de laboratoire (PlantVillage), puis nous mesurons et réduisons leur chute de performance sur des
photos de terrain (PlantDoc).

## Résultats principaux

*À compléter à la fin du projet (copier depuis `results/results.csv`).*

| Modèle | Rappel macro PlantVillage | Rappel macro PlantDoc | Rappel « malade » PlantDoc |
|---|---|---|---|
| MLP (baseline) | | | |
| CNN, PlantVillage seul | | | |
| CNN, + augmentation + fond | | | |
| CNN, fine-tuning PlantDoc | | | |
| CNN, entraînement mixte | | | |

## Données

Seules les **8 classes tomate communes aux deux jeux** sont utilisées.

| Classe unifiée | Dossier PlantVillage | Dossier PlantDoc |
|---|---|---|
| bacterial_spot | Tomato___Bacterial_spot | Tomato leaf bacterial spot |
| early_blight | Tomato___Early_blight | Tomato Early blight leaf |
| late_blight | Tomato___Late_blight | Tomato leaf late blight |
| leaf_mold | Tomato___Leaf_Mold | Tomato mold leaf |
| septoria_leaf_spot | Tomato___Septoria_leaf_spot | Tomato Septoria leaf spot |
| yellow_leaf_curl_virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato leaf yellow virus |
| mosaic_virus | Tomato___Tomato_mosaic_virus | Tomato leaf mosaic virus |
| healthy | Tomato___healthy | Tomato leaf |

Exclues : `Target_Spot` (absente de PlantDoc) et `Spider_mites` (2 images seulement dans PlantDoc).

**Découpage sans fuite** (`scripts/prepare_data.py`) :

- **PlantVillage** : 70 / 15 / 15 % stratifié et **groupé par feuille**. Plusieurs photos d'une même
  feuille partagent le même suffixe de nom de fichier (ex. `Leaf 259.1`, `Leaf 259.2`) ; elles restent
  toujours dans le même split. Un contrôle automatique vérifie qu'aucun groupe n'est partagé.
- **PlantDoc** : doublons exacts supprimés (hash MD5). Le test officiel ne contient qu'environ 70 images
  tomate ; on refait donc un tirage stratifié (25 % test, puis 20 % du reste en validation).
  `--pd-test-mode official` conserve le découpage des auteurs.
- `pd_test` n'est **jamais** utilisé pour entraîner ni pour choisir un modèle.

Sources (licence CC BY 4.0) :

- Hughes & Salathé (2015), *An open access repository of images on plant health…*, arXiv:1511.08060.
  <https://github.com/spMohanty/PlantVillage-Dataset>
- Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD.
  <https://github.com/pratikkayal/PlantDoc-Dataset>

## Exécution

### Sur Google Colab (recommandé)

1. Ouvrir `notebooks/projet_cnn_tomate.ipynb` dans Colab
   (*Fichier > Ouvrir un notebook > GitHub*, coller l'URL du dépôt).
2. *Exécution > Modifier le type d'exécution > GPU T4*.
3. Dans la première cellule, remplacer `REPO_URL` par l'URL de ce dépôt.
4. *Exécution > Tout exécuter*. Compter environ 1 h à 1 h 30 sur GPU T4.

### En local

```bash
git clone https://github.com/VOTRE_COMPTE/plant-disease-cnn.git
cd plant-disease-cnn
pip install -r requirements.txt
python scripts/download_data.py                              # ~2,4 Go dans data/raw
python scripts/prepare_data.py --pv-max-per-class 1500       # écrit data/manifest.csv
jupyter notebook notebooks/projet_cnn_tomate.ipynb
```

Options de `prepare_data.py` : `--seed`, `--pd-test-mode {resplit,official}`, `--pd-test-frac`,
`--pv-max-per-class` (limite PlantVillage pour accélérer ; sans cette option, ~15 000 images).

## Structure du dépôt

```
plant-disease-cnn/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_data.py      # PlantDoc + dossiers tomate de PlantVillage
│   └── prepare_data.py       # correspondance des classes, dédoublonnage, découpage
├── notebooks/
│   └── projet_cnn_tomate.ipynb
├── models/                   # modèles .keras (créés par le notebook, non versionnés)
├── results/                  # results.csv et figures
└── docs/                     # fiche d'impact et rapport (PDF)
```

## Protocole expérimental

| Expérience | Entraînement | Validation | Test |
|---|---|---|---|
| MLP baseline | pv_train | pv_val | pv_test, pd_test |
| CNN variantes A / B / C | pv_train | pv_val | pv_test |
| E0 : PlantVillage seul | pv_train | pv_val | pv_test, pd_test |
| E1 : + augmentation | pv_train augmenté | pv_val | pv_test, pd_test |
| E2 : + changement de fond | pv_train augmenté, fonds PlantDoc | pv_val | pv_test, pd_test |
| E3 : fine-tuning | E2 puis pd_train | pd_val | pv_test, pd_test |
| E4 : mixte | pv_train + pd_train (50/50) | pd_val | pv_test, pd_test |
| E5 : PlantDoc seul (référence) | pd_train | pd_val | pv_test, pd_test |

Métrique principale : **rappel macro**, car une maladie non détectée coûte plus cher qu'une fausse alerte.
Sont aussi rapportés la précision, le rappel et le F1 par classe, le F1 macro, le rappel « malade »
(feuilles malades non classées saines) et le taux de fausse alerte.

## Limites

Voir la section « Limites et risques » du rapport : petit test terrain (~185 images), images PlantDoc
issues du web et non de champs ouest-africains, maladies absentes du modèle, pas de classe « autre ».
