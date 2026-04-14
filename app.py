import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

models = {}
scaler = None
feature_names = None

def load_models():
    global models, scaler, feature_names
    try:
        models['rf']  = joblib.load(os.path.join(MODELS_DIR, 'model_rf_final.pkl'))
        models['xgb'] = joblib.load(os.path.join(MODELS_DIR, 'model_xgb_final.pkl'))
        models['lr']  = joblib.load(os.path.join(MODELS_DIR, 'model_lr_final.pkl'))
        scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler_lr.pkl'))
        feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
        print("✓ Modèles chargés avec succès")
    except Exception as e:
        print(f"⚠ Modèles non chargés : {e}")

load_models()

ORDINAL_LABEL_MAPS = {
    'EnvironmentSatisfaction': {1: '1_Low', 2: '2_Medium', 3: '3_High', 4: '4_Very_High'},
    'JobSatisfaction'        : {1: '1_Low', 2: '2_Medium', 3: '3_High', 4: '4_Very_High'},
    'WorkLifeBalance'        : {1: '1_Bad', 2: '2_Good',   3: '3_Better', 4: '4_Best'},
    'JobInvolvement'         : {1: '1_Low', 2: '2_Medium', 3: '3_High', 4: '4_Very_High'},
    'Education'              : {1: '1_Below_College', 2: '2_College', 3: '3_Bachelor',
                                4: '4_Master', 5: '5_Doctor'},
    'JobLevel'               : {1: '1_Junior', 2: '2_Mid_Level', 3: '3_Senior',
                                4: '4_Lead',   5: '5_Executive'},
    'PerformanceRating'      : {1: '1_Low', 2: '2_Good', 3: '3_Excellent', 4: '4_Outstanding'},
    'StockOptionLevel'       : {0: '0_None', 1: '1_Low', 2: '2_Medium', 3: '3_High'},
}
TRAVEL_MAP = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
COLS_MISSING = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
                'NumCompaniesWorked', 'TotalWorkingYears']
COLS_DROP = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID', 'Attrition']
NOMINAL_COLS = ['Gender', 'Department', 'MaritalStatus', 'JobRole', 'EducationField']
COLS_CONTINUES = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel',
]

def preprocess(df_raw):
    df = df_raw.copy()
    
    cols_to_drop = [c for c in COLS_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Features manquantes
    for col in COLS_MISSING:
        if col in df.columns and df[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)
    
    # BusinessTravel
    if 'BusinessTravel' in df.columns:
        df['BusinessTravel'] = df['BusinessTravel'].map(TRAVEL_MAP).fillna(1)
    
    ordinal_cols_encoded = []
    for col, mapping in ORDINAL_LABEL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            ordinal_cols_encoded.append(col)
    
    all_to_encode = [c for c in NOMINAL_COLS if c in df.columns] + ordinal_cols_encoded
    for col in all_to_encode:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
    
    df = pd.get_dummies(df, columns=[c for c in all_to_encode if c in df.columns],
                        drop_first=False, dtype=int)
    
    # Imputation numérique
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
    
    return df

def get_risk_level(prob):
    if prob >= 0.7:
        return 'Critique', '#FF4757'
    elif prob >= 0.5:
        return 'Élevé', '#FF6B35'
    elif prob >= 0.3:
        return 'Modéré', '#FFA502'
    else:
        return 'Faible', '#2ED573'

def generate_recommendations(row_raw, prob):
    recs = []
    
    income = row_raw.get('MonthlyIncome', 0)
    if pd.notna(income) and income < 5000:
        recs.append({'icon': '💰', 'priority': 'high', 'title': 'Révision salariale urgente',
                     'text': f'Salaire mensuel ({income:.0f}) bien en dessous du marché. Benchmarking et revalorisation recommandés.'})
    
    years_promo = row_raw.get('YearsSinceLastPromotion', 0)
    if pd.notna(years_promo) and years_promo > 3:
        recs.append({'icon': '📈', 'priority': 'high', 'title': 'Plan de carrière à définir',
                     'text': f'{years_promo:.0f} ans sans promotion. Entretien de développement à planifier rapidement.'})
    
    job_sat = row_raw.get('JobSatisfaction', 3)
    if pd.notna(job_sat) and job_sat <= 2:
        recs.append({'icon': '🎯', 'priority': 'high', 'title': 'Satisfaction au travail faible',
                     'text': 'Score de satisfaction bas. Entretien individuel urgent pour identifier les sources de mécontentement.'})
    
    wlb = row_raw.get('WorkLifeBalance', 3)
    if pd.notna(wlb) and wlb <= 2:
        recs.append({'icon': '⚖️', 'priority': 'medium', 'title': 'Équilibre vie pro/perso dégradé',
                     'text': 'Flexibilité horaire ou télétravail à envisager pour réduire le stress.'})
    
    travel = row_raw.get('BusinessTravel', '')
    if str(travel) == 'Travel_Frequently' or travel == 2:
        recs.append({'icon': '✈️', 'priority': 'medium', 'title': 'Déplacements fréquents',
                     'text': 'Réduire les déplacements non-essentiels ou compenser avec avantages supplémentaires.'})
    
    years_company = row_raw.get('YearsAtCompany', 5)
    if pd.notna(years_company) and years_company < 3:
        recs.append({'icon': '🤝', 'priority': 'medium', 'title': 'Profil junior à risque',
                     'text': 'Moins de 3 ans d\'ancienneté. Renforcer le programme de mentorat et les check-ins réguliers.'})
    
    env_sat = row_raw.get('EnvironmentSatisfaction', 3)
    if pd.notna(env_sat) and env_sat <= 2:
        recs.append({'icon': '🏢', 'priority': 'low', 'title': 'Environnement de travail insatisfaisant',
                     'text': 'Évaluer les conditions de travail physiques et relationnelles au sein de l\'équipe.'})
    
    if prob >= 0.7 and not recs:
        recs.append({'icon': '⚠️', 'priority': 'high', 'title': 'Risque de départ élevé',
                     'text': 'Risque élevé détecté par le modèle. Entretien de rétention recommandé dans les 30 jours.'})
    
    return recs


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Format CSV requis'}), 400
    
    try:
        df_raw = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Erreur lecture CSV : {str(e)}'}), 400
    
    if feature_names is None:
        return jsonify({'error': 'Modèles non chargés. Placez les fichiers .pkl dans le dossier models/'}), 500
    
    # Préserver les données brutes pour affichage
    raw_records = df_raw.copy()
    
    id_col = None
    for c in ['EmployeeID', 'Employee_ID', 'ID', 'EmployeeNumber']:
        if c in df_raw.columns:
            id_col = c
            break
    
    has_attrition = 'Attrition' in df_raw.columns
    
    try:
        df_proc = preprocess(df_raw)
    except Exception as e:
        return jsonify({'error': f'Erreur prétraitement : {str(e)}'}), 500
    
    try:
        model = models['rf']
        probs = model.predict_proba(df_proc)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        return jsonify({'error': f'Erreur prédiction : {str(e)}'}), 500
    
    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        row = raw_records.iloc[i].to_dict()
        risk_label, risk_color = get_risk_level(prob)
        
        emp_id = row.get(id_col, i + 1) if id_col else i + 1
        
        actual = None
        if has_attrition:
            actual = row.get('Attrition', None)
            if actual in ['Yes', 'yes', '1', 1, True]:
                actual = 1
            elif actual in ['No', 'no', '0', 0, False]:
                actual = 0
        
        recs = generate_recommendations(row, float(prob))
        
        results.append({
            'id': str(emp_id),
            'index': i,
            'prob': round(float(prob) * 100, 1),
            'pred': int(pred),
            'risk_label': risk_label,
            'risk_color': risk_color,
            'actual': actual,
            'recs': recs,
            'data': {k: (None if (isinstance(v, float) and np.isnan(v)) else v) 
                     for k, v in row.items()
                     if k not in ['EmployeeCount', 'Over18', 'StandardHours']}
        })
    
    # Stats globales
    n = len(results)
    n_risk = sum(1 for r in results if r['prob'] >= 50)
    n_critical = sum(1 for r in results if r['prob'] >= 70)
    avg_prob = round(float(np.mean(probs)) * 100, 1)
    
    dept_stats = {}
    if 'Department' in df_raw.columns:
        for _, row in df_raw.iterrows():
            dept = str(row.get('Department', 'Unknown'))
            idx = list(df_raw.index).index(row.name) if row.name in df_raw.index else 0
            if dept not in dept_stats:
                dept_stats[dept] = {'total': 0, 'at_risk': 0}
            dept_stats[dept]['total'] += 1
            if probs[list(df_raw.index).index(row.name)] >= 0.5:
                dept_stats[dept]['at_risk'] += 1
    
    accuracy = None
    if has_attrition:
        actuals = [r['actual'] for r in results if r['actual'] is not None]
        preds_l = [r['pred'] for r in results if r['actual'] is not None]
        if actuals:
            correct = sum(1 for a, p in zip(actuals, preds_l) if a == p)
            accuracy = round(correct / len(actuals) * 100, 1)
    
    return jsonify({
        'total': n,
        'at_risk': n_risk,
        'critical': n_critical,
        'avg_prob': avg_prob,
        'accuracy': accuracy,
        'dept_stats': dept_stats,
        'results': results,
        'columns': list(df_raw.columns),
        'has_attrition': has_attrition,
        'prob_distribution': [round(float(p) * 100, 1) for p in sorted(probs, reverse=True)]
    })

@app.route('/api/models_status')
def models_status():
    return jsonify({
        'loaded': len(models) > 0,
        'models': list(models.keys()),
        'features': len(feature_names) if feature_names else 0
    })

if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
