# HumanForYou — Attrition Intelligence App

Interface IA sobre pour visualiser les prédictions d'attrition des employés.

## Installation

```bash
pip install flask pandas numpy scikit-learn xgboost joblib
```

## Structure attendue

```
hr_attrition_app/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── models/
    ├── model_rf_final.pkl      ← copier depuis votre notebook
    ├── model_xgb_final.pkl
    ├── model_lr_final.pkl
    ├── scaler_lr.pkl
    └── feature_names.pkl
```

## Étape 1 — Copier les modèles

Après avoir exécuté votre notebook, copiez les fichiers `.pkl` dans le dossier `models/` :

```bash
cp model_rf_final.pkl  hr_attrition_app/models/
cp model_xgb_final.pkl hr_attrition_app/models/
cp model_lr_final.pkl  hr_attrition_app/models/
cp scaler_lr.pkl       hr_attrition_app/models/
cp feature_names.pkl   hr_attrition_app/models/
```

## Étape 2 — Lancer l'application

```bash
cd hr_attrition_app
python app.py
```

Puis ouvrir : **http://localhost:5000**

## Utilisation

1. Glisser-déposer votre fichier CSV sur la page d'accueil
2. Le dashboard affiche automatiquement les statistiques globales
3. Onglet **Employés** : tableau filtrable et triable, cliquer sur une ligne pour le détail
4. Onglet **Graphiques** : distributions, analyses par département, salaire vs risque
5. Le panneau employé détaille : score de risque, données brutes, **recommandations RH personnalisées**

## Format CSV attendu

Le CSV doit contenir les colonnes RH du projet HumanForYou :
`Age, Department, JobRole, MonthlyIncome, JobSatisfaction, WorkLifeBalance,
EnvironmentSatisfaction, BusinessTravel, YearsAtCompany, YearsSinceLastPromotion…`

La colonne `Attrition` est optionnelle — si présente, la précision du modèle est affichée.

---

## Notebook — HumanForYou_ultimateSpiderMan_v3.ipynb

Le notebook couvre l'intégralité du pipeline ML ayant produit les modèles utilisés par l'application.

### Structure du notebook

| Étape | Contenu |
|---|---|
| 1. EDA | Exploration, distribution de l'attrition, corrélations, boxplots |
| 2. Prétraitement | Fusion des 4 sources, encodage, indicateurs de valeurs manquantes, imputation KNN |
| 3. Split train/test | Séparation stratifiée 80/20, normalisation ciblée sur les variables continues |
| 4. Modèles | Régression Logistique, Random Forest (GridSearchCV), XGBoost (GridSearchCV) |
| 5. Évaluation | Tableau comparatif, courbes ROC, courbes d'apprentissage |
| 6. Interprétabilité | Feature importance Gini, SHAP summary plot, SHAP waterfall individuel |
| 7. Recommandations | Employés à risque, facteurs clés, tableau d'actions RH priorisées |

### Données d'entrée

Le notebook attend 4 fichiers CSV dans le répertoire de travail :

```
general_data.csv
manager_survey_data.csv
employee_survey_data.csv
in_out_time/
    in_time.csv
    out_time.csv        ← optionnel (features horaires)
```

### Modèles produits

À la fin de l'exécution, le notebook exporte :

```
model_rf_final.pkl      ← Random Forest 
model_xgb_final.pkl
model_lr_final.pkl
scaler_lr.pkl           ← StandardScaler ajusté sur le train
feature_names.pkl       ← liste ordonnée des colonnes après encodage
```

Ces fichiers sont directement utilisables par `hr_attrition_app/`.

### Choix techniques

- Métrique d'optimisation : **Recall** sur la classe attrition (priorité sur les faux négatifs)
- Déséquilibre de classes géré via `class_weight='balanced'` (LR, RF) et `scale_pos_weight` (XGBoost)
- Imputation **après le split** pour éviter le data leakage (KNNImputer, n_neighbors=5)
- Indicateurs `_missing` créés avant l'imputation pour capturer le signal de non-réponse
- Encodage différencié : ordinale technique (entier), ordinale sentiment (One-Hot), nominale (One-Hot)

### Dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
```
