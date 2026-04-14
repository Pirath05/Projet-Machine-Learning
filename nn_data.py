import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def load_data():
    general      = pd.read_csv('general_data.csv')
    manager_surv = pd.read_csv('manager_survey_data.csv')
    emp_surv     = pd.read_csv('employee_survey_data.csv')

    df = general.merge(manager_surv, on='EmployeeID', how='left')
    df = df.merge(emp_surv,          on='EmployeeID', how='left')

    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID'],
            inplace=True, errors='ignore')

    for col in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values, scaler, X.columns.tolist()