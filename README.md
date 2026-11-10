# SenHub-Deep-Learningplant-disease-cnn
# Détection des maladies de la tomate par CNN : du laboratoire au champ

Projet de fin de module Deep Learning, cohorte 3 AI Hub — Dakar Institute of Technology (DIT), 2026.

- **Groupe** : DJIGUIMDE Oumaro Titans et DIOUF Ousseynou
- **Encadrant** : M. Amadou Alwaly NDIAYE
- **Framework** : Keras / TensorFlow
- **Livrables** : [rapport technique](docs/https://github.com/Odjiguimde/SenHub-Deep-Learningplant-disease-cnn/blob/main/docs/Rapport_technique_CNN.pdf) · [fiche d'impact](https://github.com/Odjiguimde/SenHub-Deep-Learningplant-disease-cnn/blob/main/docs/Fiche_impact.pdf) · [notebook](notebooks/https://github.com/Odjiguimde/SenHub-Deep-Learningplant-disease-cnn/blob/main/notebooks/projet_cnn_tomate.ipynb)

Une coopérative agricole veut qu'un agriculteur puisse photographier une feuille de tomate avec son
smartphone et obtenir un diagnostic préliminaire. Nous entraînons des CNN conçus **from scratch** sur des
images de laboratoire (PlantVillage), puis nous mesurons et réduisons leur chute de performance sur des
photos de terrain (PlantDoc).

> **Question de recherche** : de combien chute un CNN entraîné en laboratoire quand on le teste sur le
> terrain, et quelles stratégies réduisent cet écart sans modèle pré-entraîné ?

## En bref

- Un CNN léger (≈ 289 000 paramètres) atteint **98,3 % de rappel macro** sur PlantVillage…
- …mais seulement **18,4 %** sur les photos de terrain PlantDoc, à peine au-dessus du hasard (12,5 % pour 8 classes).
- La meilleure stratégie testée, l'**entraînement mixte** PlantVillage + PlantDoc, remonte à **31,4 %** (2,5 fois le hasard, +13 points).
- **Le prototype n'est pas prêt pour un déploiement auprès des agriculteurs.**

## Résultats principaux

Rappel macro (métrique principale), sur `pv_test` (1 405 images) et `pd_test` (185 images).

| Modèle | Rappel macro PlantVillage | Rappel macro PlantDoc | Rappel « malade » PlantDoc |
|---|---|---|---|
| MLP (baseline) | 13,5 % | 11,6 % | — |
| CNN, PlantVillage seul (E0) | 98,3 % | 18,4 % | 93,5 % |
| CNN, + augmentation + fond (E2) | 73,8 % | 15,7 % | 76,3 % |
| CNN, fine-tuning PlantDoc (E3) | 66,2 % | 23,0 % | 63,9 % |
| CNN, entraînement mixte (E4) | 65,6 % | **31,4 %** | 91,7 % |

### Comparaison MLP / variantes CNN (laboratoire)

| Modèle | Paramètres | Rappel macro pv_test | F1 macro pv_test |
|---|---|---|---|
| MLP (64×64 aplaties, Dense 512 → 256) | 6 425 352 | 13,5 % | 5,2 % |
| CNN A : 3 blocs, sans batch norm | 288 040 | 96,0 % | 95,5 % |
| **CNN B : 3 blocs + batch norm (retenu)** | 289 384 | 98,3 % | 98,1 % |
| CNN C : 4 blocs + batch norm | 1 177 192 | 98,7 % | 98,7 % |

- Le **MLP n'apprend pas** : sa perte reste bloquée vers 2,05 (≈ ln 8, le hasard) et il prédit presque toujours « septoriose ». En aplatissant l'image, il perd la structure spatiale, avec 22 fois plus de paramètres qu'un CNN.
- La **batch normalization** apporte +2,3 points.
- Le **4ᵉ bloc** n'apporte que +0,4 point pour 4 fois plus de paramètres : nous retenons la **variante B**, plus légère pour un smartphone.

### Laboratoire → terrain (toutes les expériences)

| Exp. | Rappel macro pv_test | Rappel macro pd_test | Chute (points) | F1 macro pd_test | Rappel « malade » pd_test | Fausse alerte pd_test |
|---|---|---|---|---|---|---|
| E0 PlantVillage seul | 98,3 % | 18,4 % | 79,9 | 17,3 % | 93,5 % | 93,8 % |
| E1 + augmentation | 85,8 % | 18,2 % | 67,6 | 15,4 % | 91,7 % | 93,8 % |
| E2 + changement de fond | 73,8 % | 15,7 % | 58,1 | 13,4 % | 76,3 % | 75,0 % |
| E3 fine-tuning PlantDoc | 66,2 % | 23,0 % | 43,2 | 17,7 % | 63,9 % | 18,8 % |
| **E4 mixte** | 65,6 % | **31,4 %** | 34,2 | **30,3 %** | 91,7 % | 75,0 % |
| E5 PlantDoc seul | 12,3 % | 16,0 % | — | 5,2 % | 84,6 % | 56,2 % |

Ce qu'il faut retenir :

- **E0 — le fossé est massif** : 98,3 % → 18,4 %. Le rappel « malade » de 93,5 % est trompeur : le modèle déclare presque tout malade, y compris 94 % des feuilles saines.
- **E1 — l'augmentation classique ne suffit pas** : aucun gain sur le terrain, 12,5 points perdus en laboratoire.
- **E2 — le changement de fond a échoué à cause de notre implémentation** : les fonds collés, tirés de PlantDoc, contiennent déjà des feuilles malades d'autres classes (bruit d'étiquette). Le modèle sous-apprend (72 % d'accuracy d'entraînement). Ce résultat ne réfute pas l'idée.
- **E3 — le fine-tuning** gagne 4,6 points mais déclare saines 36 % des feuilles malades, l'erreur la plus coûteuse. 444 images de terrain ne suffisent pas.
- **E4 — l'entraînement mixte est la meilleure stratégie** : 31,4 %. Il reconnaît le mieux le virus des feuilles jaunes (58 %) et le mildiou (54 %), très mal la cladosporiose et l'alternariose (9 %).
- **E5 — PlantVillage reste utile** : entraîné sur PlantDoc seul, le CNN s'effondre (16,0 %).

> ⚠ Avec 185 images de test et une seule graine, les écarts entre E0, E1, E2 et E5 (16 à 18 %) ne sont
> pas significatifs. L'avance de E4 (+13 points) est plus nette, mais devrait être confirmée avec
> plusieurs graines.

### Analyse des erreurs et Grad-CAM

- **11 des 12 erreurs les plus confiantes** du modèle fine-tuné (E3) sont des maladies classées « saines » : sur le terrain, la colonne « sain » absorbe 30 à 63 % de chaque maladie. Le modèle semble avoir appris un raccourci : « photo de terrain, verte et chargée = saine ».
- Causes récurrentes : plusieurs feuilles par image, lésions trop petites après redimensionnement en 128 × 128, fonds inédits, mains, filigranes (« alamy stock photo »).
- Les confusions entre maladies vont surtout vers la **septoriose**, visuellement proche de l'alternariose et de la tache bactérienne.
- **Grad-CAM** : en laboratoire, la part de l'attention sur la feuille passe de 59 % (E0) à 69 % (E2) pour une feuille qui occupe 43 % de l'image. Sur le terrain, l'attention se disperse (tiges, mains, bords, fond) : la concentration gagnée en laboratoire ne se transfère pas.

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

### Effectifs par split

Exécution Colab avec `--pv-max-per-class 1500` et graine 42 : 9 825 images PlantVillage et 740 PlantDoc
(15 820 images tomate au total après nettoyage, sans limite).

| Classe | pv_train | pv_val | pv_test | pd_train | pd_val | pd_test |
|---|---|---|---|---|---|---|
| bacterial_spot | 1 074 | 214 | 212 | 66 | 16 | 27 |
| early_blight | 716 | 142 | 142 | 53 | 13 | 22 |
| late_blight | 1 065 | 217 | 218 | 66 | 17 | 28 |
| leaf_mold | 682 | 135 | 135 | 54 | 14 | 23 |
| septoria_leaf_spot | 1 072 | 215 | 213 | 89 | 23 | 37 |
| yellow_leaf_curl_virus | 1 070 | 213 | 217 | 45 | 11 | 19 |
| mosaic_virus | 266 | 53 | 54 | 33 | 8 | 13 |
| healthy | 1 066 | 220 | 214 | 38 | 9 | 16 |
| **Total** | **7 011** | **1 409** | **1 405** | **444** | **111** | **185** |

Observations : PlantVillage est environ 13 fois plus grand que PlantDoc ; la mosaïque est la classe rare
des deux côtés. Les images PlantVillage font toutes 256 × 256 px (feuille seule, fond gris uniforme) ;
celles de PlantDoc vont de 130 à 4 320 px de large et montrent souvent plusieurs feuilles, des mains, de
la terre ou des filigranes.

### Découpage sans fuite (`scripts/prepare_data.py`)

- **PlantVillage** : 70 / 15 / 15 % stratifié et **groupé par feuille**. Plusieurs photos d'une même
  feuille partagent le même suffixe de nom de fichier (ex. `Leaf 259.1`, `Leaf 259.2`) ; elles restent
  toujours dans le même split. Un contrôle automatique vérifie qu'aucun groupe n'est partagé.
- **PlantDoc** : 4 doublons exacts supprimés (hash MD5). Le test officiel ne contient qu'environ 70 images
  tomate ; on refait donc un tirage stratifié (25 % test, puis 20 % du reste en validation).
  `--pd-test-mode official` conserve le découpage des auteurs.
- `pd_test` n'est **jamais** utilisé pour entraîner ni pour choisir un modèle.

### Sources (licence CC BY 4.0)

- Hughes & Salathé (2015), *An open access repository of images on plant health to enable the development
  of mobile disease diagnostics*, arXiv:1511.08060. <https://github.com/spMohanty/PlantVillage-Dataset>
- Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD, p. 249-253.
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
git clone https://github.com/<VOTRE_COMPTE>/SenHub-Deep-Learningplant-disease-cnn.git
cd SenHub-Deep-Learningplant-disease-cnn
pip install -r requirements.txt
python scripts/download_data.py                              # ~2,4 Go dans data/raw
python scripts/prepare_data.py --pv-max-per-class 1500       # écrit data/manifest.csv
jupyter notebook notebooks/projet_cnn_tomate.ipynb
```

Options de `prepare_data.py` : `--seed` (42 par défaut dans nos résultats), `--pd-test-mode {resplit,official}`,
`--pd-test-frac`, `--pv-max-per-class` (limite PlantVillage pour accélérer ; sans cette option, ~15 000 images).

## Structure du dépôt

```
SenHub-Deep-Learningplant-disease-cnn/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_data.py      # PlantDoc + dossiers tomate de PlantVillage
│   └── prepare_data.py       # correspondance des classes, dédoublonnage, découpage
├── notebooks/
│   └── projet_cnn_tomate.ipynb
├── models/                   # modèles .keras (créés par le notebook, non versionnés)
├── results/                  # results.csv, matrices de confusion, courbes, Grad-CAM
└── docs/
    ├── Rapport_technique_CNN.pdf
    └── Fiche_impact.pdf
```

## Protocole expérimental

### Modèles (entrées 128 × 128 × 3)

| Modèle | Architecture | Rôle |
|---|---|---|
| MLP | images 64 × 64 aplaties, Dense 512 → 256, dropout 0,5 | baseline sans structure spatiale |
| CNN A | 3 blocs (2 × Conv 3×3 + MaxPool), 32-64-128 filtres, sans BN, GAP, dropout | variante de référence |
| CNN B | comme A, avec batch normalization | effet de la batch norm |
| CNN C | comme B, 4 blocs (jusqu'à 256 filtres) | effet de la profondeur |

**Entraînement** : Adam (lr 10⁻³), entropie croisée, poids de classes équilibrés, early stopping sur la
perte de validation (patience 6), réduction du learning rate sur plateau.

### Expériences de robustesse (avec CNN B)

| Expérience | Entraînement | Validation | Test | Idée testée |
|---|---|---|---|---|
| MLP baseline | pv_train | pv_val | pv_test, pd_test | intérêt de la convolution |
| CNN variantes A / B / C | pv_train | pv_val | pv_test | batch norm, profondeur |
| E0 : PlantVillage seul | pv_train | pv_val | pv_test, pd_test | mesure de la chute labo → terrain |
| E1 : + augmentation | pv_train augmenté (rotation, retournement, recadrage, luminosité, contraste) | pv_val | pv_test, pd_test | variations photométriques et géométriques |
| E2 : + changement de fond | pv_train augmenté, fonds PlantDoc | pv_val | pv_test, pd_test | casser le raccourci « fond uniforme » |
| E3 : fine-tuning | E2 puis pd_train (lr 10⁻⁴) | pd_val | pv_test, pd_test | adaptation avec peu de données terrain |
| E4 : mixte | pv_train + pd_train (50/50) | pd_val | pv_test, pd_test | entraînement mixte |
| E5 : PlantDoc seul (référence) | pd_train | pd_val | pv_test, pd_test | PlantVillage aide-t-il ? |

**Changement de fond (E2)** : un masque de feuille approximatif (saturation HSV > 0,25, lissé) sépare la
feuille du fond gris de PlantVillage ; le fond est remplacé, avec une probabilité de 0,6, par une image
de `pd_train` (étiquettes PlantDoc non utilisées).

### Métriques

Métrique principale : **rappel macro**, car une maladie non détectée coûte plus cher qu'une fausse alerte.
Sont aussi rapportés la précision, le rappel et le F1 par classe, le F1 macro, le rappel « malade »
(feuilles malades non classées saines) et le taux de fausse alerte (feuilles saines classées malades).
L'accuracy n'est donnée qu'à titre indicatif.

## Impact pour l'agriculteur

Résumé de la [fiche d'impact](docs/Fiche_impact.pdf).

- **Utilisateur** : maraîcher de la coopérative, sans formation en phytopathologie ni agronome à proximité,
  avec un smartphone d'entrée ou de milieu de gamme, souvent sans connexion → modèle léger, idéalement hors ligne.
- **Pourquoi la tomate** : culture la mieux couverte par les deux jeux (8 classes communes) ; deuxième culture
  maraîchère au Sénégal (environ 22,5 % de la production de légumes) ; elle mêle maladies fongiques
  (traitables) et virales (sans traitement curatif), donc des coûts d'erreur très différents.
- **Coût des erreurs** :

| Erreur | Conséquence | Gravité |
|---|---|---|
| Feuille malade classée « saine » | aucune action ; la maladie gagne la parcelle, perte de récolte possible | élevée |
| Maladie confondue avec une autre | produit inadapté (fongicide contre un virus), argent perdu | moyenne à élevée |
| Feuille saine classée malade | traitement inutile : coût, exposition aux pesticides, résidus | modérée |

- **Ce que l'outil pourrait changer** : un premier avis en quelques secondes au champ, des traitements mieux
  ciblés, savoir quand appeler un conseiller, et des alertes remontées à la coopérative par zone.

## Limites et risques

**Le prototype n'est pas prêt pour un déploiement** : il n'a jamais été testé sur des photos prises dans
les champs de la coopérative.

- **Données** : PlantDoc provient d'images du web, pas de champs ouest-africains ; petit test terrain
  (185 images, 13 à 37 par classe) ; erreurs d'étiquetage possibles ; seules 7 maladies couvertes et pas
  de classe « autre » (flétrissement bactérien, fusariose, ravageurs ou carences seront mal classés).
- **Méthodes** : masque de feuille par seuil de saturation, qui échoue sur les feuilles pâles ou jaunies ;
  erreur de conception de E2 (fonds contenant des feuilles malades) ; une seule graine par expérience,
  donc les écarts de moins de 5 points ne sont pas interprétables ; CNN from scratch uniquement, comme
  le demandait l'énoncé.
- **Risques** : faux « sain » (maladie non traitée), mauvais diagnostic (produit inadapté, dépense,
  exposition aux pesticides), excès de confiance dans l'outil au détriment du conseiller agricole.

## Pistes d'amélioration

- Afficher la confiance et répondre « incertain, consultez un conseiller » sous un seuil.
- Ajouter une classe « autre / hors sujet » (images qui ne sont pas des feuilles de tomate).
- Refaire E2 avec des fonds sans tomate.
- Collecter et étiqueter des photos locales avec des agronomes, puis réentraîner.
- Transfer learning (MobileNetV2, EfficientNet) pour un modèle léger et plus robuste.
- Évaluer sur plusieurs graines et en validation croisée sur PlantDoc ; analyser en détail le modèle E4.

## Références

- Hughes, D. P. et Salathé, M. (2015). *An open access repository of images on plant health to enable the
  development of mobile disease diagnostics*. arXiv:1511.08060.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S. et Batra, N. (2020). *PlantDoc: A Dataset for Visual
  Plant Disease Detection*. CoDS-COMAD 2020, p. 249-253.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization*. ICCV 2017.
- Ioffe, S. et Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift*. ICML 2015.

## Licence des données

PlantVillage et PlantDoc sont distribués sous licence Creative Commons Attribution 4.0 (CC BY 4.0).
