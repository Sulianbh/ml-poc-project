# Dossier `plots/` — Visualisations exportées

Ce dossier contient les 11 graphiques exportés par les notebooks Jupyter.
Ils constituent la **trace visuelle** de l'analyse : chaque graphique correspond
à une décision ou une observation documentée dans les notebooks.

---

## Contenu du dossier

| Fichier                              | Taille | Généré par              | Thème                         |
|--------------------------------------|--------|-------------------------|-------------------------------|
| `01_distribution_puissance.png`      | 30 Ko  | `01_exploration.ipynb`  | Exploration des données       |
| `02_types_connecteurs.png`           | 35 Ko  | `01_exploration.ipynb`  | Exploration des données       |
| `03_implantation.png`                | 50 Ko  | `01_exploration.ipynb`  | Exploration des données       |
| `04_condition_acces.png`             | 23 Ko  | `01_exploration.ipynb`  | Exploration des données       |
| `05_carte_allego.png`                | 45 Ko  | `01_exploration.ipynb`  | Exploration des données       |
| `06_distribution_nb_pdc.png`         | 39 Ko  | `02_clustering.ipynb`   | Clustering & labellisation    |
| `07_elbow_silhouette.png`            | 71 Ko  | `02_clustering.ipynb`   | Clustering & labellisation    |
| `08_carte_clusters.png`              | 78 Ko  | `02_clustering.ipynb`   | Clustering & labellisation    |
| `09_comparaison_modeles.png`         | 38 Ko  | `03_modelisation.ipynb` | Évaluation des modèles        |
| `10_matrices_confusion.png`          | 65 Ko  | `03_modelisation.ipynb` | Évaluation des modèles        |
| `11_feature_importance_xgboost.png`  | 47 Ko  | `03_modelisation.ipynb` | Évaluation des modèles        |

> Ces fichiers sont générés automatiquement à chaque exécution des notebooks.
> Ils ne doivent **jamais** être modifiés à la main.

---

## Phase 1 — Exploration des données (`01_exploration.ipynb`)

### `01_distribution_puissance.png`

**Type :** histogramme
**Données :** colonne `puissance_nominale` des bornes Allego (en kW)

Montre la répartition des puissances nominales sur l'ensemble du réseau Allego.
On observe plusieurs **pics nets** correspondant aux standards de recharge rapide :
50 kW (bornes de première génération), 150 kW et 350 kW (bornes haute puissance).
La moyenne est d'environ 100 kW, confirmant la spécialisation d'Allego dans la
recharge rapide (DC).

**Ce qu'on apprend :** la distribution est **multimodale** (plusieurs modes) —
ce n'est pas une courbe normale. Les modèles ne doivent pas supposer une distribution
gaussienne pour cette feature.

---

### `02_types_connecteurs.png`

**Type :** graphique en barres
**Données :** proportion de bornes équipées de chaque type de prise

Affiche le nombre de points de charge (PDC) disposant de chaque type de connecteur :
Type EF (prise domestique), Type 2 (Mennekes AC), CCS Combo (DC rapide),
CHAdeMO (DC rapide, standard japonais), Autre.

**Ce qu'on apprend :** la prise **CCS Combo** équipe 62 % des bornes Allego —
c'est le standard européen pour la recharge rapide DC, dominant sur les réseaux modernes.
Le CHAdeMO (5.7 %) est en déclin. La prise "Autre" est quasi-absente.

---

### `03_implantation.png`

**Type :** camembert (diagramme circulaire)
**Données :** colonne `implantation_station` — type d'emplacement de chaque borne

Montre la répartition par type d'emplacement : station dédiée à la recharge rapide,
parking privé à usage public, voirie, parking public, parking privé clientèle.

**Ce qu'on apprend :** la majorité des bornes Allego sont dans des **stations dédiées**
ou des **parkings privés à usage public** — cohérent avec leur modèle de superchargeurs
sur autoroutes et zones commerciales.

---

### `04_condition_acces.png`

**Type :** graphique en barres horizontales
**Données :** colonne `condition_acces` — accès libre ou restreint

Compare le nombre de bornes en accès libre versus restreint (abonnement requis).

**Ce qu'on apprend :** **100 %** des bornes Allego dans ce dataset sont en accès libre.
C'est pourquoi la feature `acces_libre` a une variance nulle dans le dataset Allego
et contribue peu à la prédiction.

---

### `05_carte_allego.png`

**Type :** nuage de points géographiques (lat/lon)
**Données :** colonnes `latitude` et `longitude` de chaque borne

Affiche la localisation de toutes les bornes Allego sur le territoire français
(France métropolitaine + DOM). Chaque point est une borne.

**Ce qu'on apprend :** les bornes Allego suivent clairement les **grands axes routiers**
(autoroutes A1, A6, A7, A10…) et se concentrent autour des **métropoles**
(Paris, Lyon, Marseille, Bordeaux, Lille). Les zones rurales sont quasi-absentes.
Cette distribution géographique est directement liée aux labels K-Means.

---

## Phase 2 — Clustering & labellisation (`02_clustering.ipynb`)

### `06_distribution_nb_pdc.png`

**Type :** deux histogrammes côte à côte
**Données :** nombre de PDC par commune — brut (gauche) vs. après `log1p` (droite)

Justifie la transformation logarithmique appliquée avant le clustering K-Means.

**Ce qu'on apprend :** la distribution brute est très **asymétrique à droite**
(quelques communes ont 50–100 bornes, la grande majorité en a moins de 10).
Après `log1p`, la distribution est beaucoup plus **symétrique et centrée** —
K-Means peut alors travailler sur des distances équilibrées sans être dominé
par les quelques communes très denses.

---

### `07_elbow_silhouette.png`

**Type :** deux courbes côte à côte
**Données :** métriques K-Means testées pour k de 2 à 8

**Graphique de gauche — Courbe du coude (Elbow Method) :**
Trace l'**inertie** (somme des distances au carré entre chaque point et son centroïde)
en fonction du nombre de clusters k. Un coude prononcé signale le k optimal :
au-delà, ajouter des clusters apporte peu de gain.

**Graphique de droite — Score de silhouette :**
Le score de silhouette mesure à quel point chaque point est bien assigné à son cluster
(1.0 = parfaitement séparé, 0.0 = sur la frontière, -1.0 = mal assigné).
Un score élevé confirme que les clusters sont compacts et bien séparés.

**Ce qu'on apprend :** les deux courbes convergent vers **k=3** comme choix optimal.
La ligne verticale rouge à k=3 matérialise ce choix dans les deux graphiques.
Le score de silhouette pour k=3 est documenté dans le notebook.

---

### `08_carte_clusters.png`

**Type :** nuage de points géographiques coloré par cluster
**Données :** coordonnées GPS moyennes de chaque commune + label K-Means

Affiche les 294 communes Allego sur la carte de France, colorées selon leur
niveau d'équipement :
- **Rouge** → Sous-équipé (classe 0 — communes isolées, peu de bornes)
- **Orange** → Normalement équipé (classe 1 — densité moyenne)
- **Vert** → Bien équipé (classe 2 — hubs importants, grandes villes)

**Ce qu'on apprend :** la séparation géographique entre les classes est **très visible**
à l'œil nu. Les communes "Bien équipé" (vert) coïncident avec les grandes métropoles
et les nœuds autoroutiers. C'est précisément ce signal géographique que XGBoost
capturera via les features `latitude` et `longitude`.

---

## Phase 3 — Évaluation des modèles (`03_modelisation.ipynb`)

### `09_comparaison_modeles.png`

**Type :** graphique en barres groupées
**Données :** métriques des 3 modèles sur le jeu de test (1 138 bornes)

Compare côte à côte les trois métriques principales pour chaque modèle :
- **Accuracy** (bleu) : proportion globale de bornes bien classifiées
- **F1 weighted** (rouge) : équilibre précision/rappel, pondéré par la taille des classes
- **F1 macro** (vert) : équilibre précision/rappel, équitable entre les 3 classes

Les valeurs sont affichées directement sur chaque barre.

**Ce qu'on apprend :** la progression est nette — LR (~60 %) → KNN (~74 %) → XGBoost (~99.6 %).
L'écart entre F1 weighted et F1 macro est faible pour tous les modèles, ce qui indique
que même la **classe minoritaire** (Sous-équipé, 8.7 %) est bien traitée.

---

### `10_matrices_confusion.png`

**Type :** trois matrices de confusion côte à côte (heatmaps)
**Données :** prédictions des 3 modèles vs. vraies étiquettes du jeu de test

Chaque matrice est un tableau 3×3 où :
- **Lignes** = vraies classes (ce que la borne est réellement)
- **Colonnes** = classes prédites (ce que le modèle a dit)
- **Diagonale** = prédictions correctes (en bleu foncé = beaucoup)
- **Hors-diagonale** = erreurs (en bleu clair = peu)

| Classe prédite | Confusion principale (LR et KNN)                              |
|----------------|---------------------------------------------------------------|
| Sous-équipé (0)| Confondu avec Normalement équipé (1) — classes adjacentes    |
| Normal (1)     | Confondu dans les deux sens avec Sous-équipé et Bien équipé  |
| Bien équipé (2)| Confondu avec Normalement équipé (1) — classes adjacentes    |

Pour XGBoost : la diagonale est quasi-parfaite, les cases hors-diagonale sont
quasi-vides — moins de 5 erreurs sur 1 138 bornes testées.

**Ce qu'on apprend :** les erreurs des modèles "faibles" (LR, KNN) concernent
principalement des **classes adjacentes** (0↔1 et 1↔2), jamais des sauts extrêmes
(0 prédit comme 2). C'est cohérent avec la nature ordinale des labels.

---

### `11_feature_importance_xgboost.png`

**Type :** graphique en barres horizontales
**Données :** score d'importance de chaque feature pour le modèle XGBoost

Affiche le **gain** de chaque feature : à quel point elle réduit l'erreur de
classification quand XGBoost l'utilise pour faire une coupure dans un arbre.
Les barres `latitude` et `longitude` sont colorées en **rouge** pour signaler
visuellement l'artefact du projet.

**Ce qu'on apprend :** `latitude` et `longitude` dominent massivement l'importance,
loin devant `puissance_nominale` ou les types de prises. XGBoost a appris à
**reconstruire la géographie des communes** plutôt qu'à utiliser les caractéristiques
techniques des bornes. C'est la preuve visuelle de l'artefact expliqué dans
`results/LISEZ-MOI.md` et `models/LISEZ-MOI.md`.

---

## Régénérer les graphiques

Les graphiques sont produits par les notebooks — il suffit de les ré-exécuter :

```bash
# Lancer Jupyter et exécuter les notebooks dans l'ordre
jupyter notebook notebooks/

# Ou en ligne de commande (exécution non-interactive)
jupyter nbconvert --to notebook --execute notebooks/01_exploration.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_modelisation.ipynb
```

> ⚠️  `03_modelisation.ipynb` nécessite que `scripts/train.py` ait été exécuté
>     au préalable (les fichiers `.joblib` doivent exister dans `models/`).

Les fichiers PNG dans ce dossier sont **écrasés** à chaque exécution des notebooks.
