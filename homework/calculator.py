import os
import gzip
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ============================================================
# RUTAS ROBUSTAS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "files", "input")
MODELS_DIR = os.path.join(BASE_DIR, "files", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "files", "output")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train_data.csv.zip")
test_path = os.path.join(INPUT_DIR, "test_data.csv.zip")

# ============================================================
# CARGA DE DATOS
# ============================================================

train_data = pd.read_csv(train_path, index_col=False, compression="zip")
test_data = pd.read_csv(test_path, index_col=False, compression="zip")

# ============================================================
# LIMPIEZA
# ============================================================

def limpiar(df):
    df = df.rename(columns={'default payment next month': 'default'})
    df.drop('ID', axis=1, inplace=True)
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    df = df.query('MARRIAGE > 0 and EDUCATION > 0')
    df = df.dropna()
    return df

train_data = limpiar(train_data)
test_data = limpiar(test_data)

x_train = train_data.drop(columns=["default"])
y_train = train_data["default"]

x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]

# ============================================================
# PIPELINE
# ============================================================

cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(dtype=int), cat_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('transformer', transformer),
    ('clasi', RandomForestClassifier(n_jobs=-1, random_state=17))
])

pipeline.fit(x_train, y_train)
print("Precisión inicial:", pipeline.score(x_test, y_test))

# ============================================================
# GRID SEARCH
# ============================================================

param_grid = {
    'clasi__n_estimators': [180],
    'clasi__max_features': ['sqrt'],
    'clasi__min_samples_split': [10],
    'clasi__min_samples_leaf': [2],
    'clasi__bootstrap': [True],
    'clasi__max_depth': [None]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True,
    verbose=True
)

grid_search.fit(x_train, y_train)

# ============================================================
# GUARDAR MODELO
# ============================================================

model_path = os.path.join(MODELS_DIR, "model.pkl.gz")

with gzip.open(model_path, 'wb') as file:
    pickle.dump(grid_search, file)

# ============================================================
# FUNCIÓN PARA CARGAR MODELO Y PREDECIR
# ============================================================

def cargar_modelo_y_predecir(data, modelo_path=model_path):
    try:
        with gzip.open(modelo_path, "rb") as file:
            estimator = pickle.load(file)
        return estimator.predict(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {modelo_path}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo o predecir: {e}")

# Predicciones
y_train_pred = cargar_modelo_y_predecir(x_train)
y_test_pred = cargar_modelo_y_predecir(x_test)

# ============================================================
# GUARDADO DE MÉTRICAS
# ============================================================

metrics_file = os.path.join(OUTPUT_DIR, "metrics.json")

def write_metric(json_dict):
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_dict) + "\n")

def evaluacion(dataset, y_true, y_pred):
    metrics = {
        "type": "metrics",
        "dataset": dataset,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred))
    }

    write_metric(metrics)

def matriz_confusion(dataset, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    
    cm = {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(matrix[0,0]),
            "predicted_1": int(matrix[0,1])
        },
        "true_1": {
            "predicted_0": int(matrix[1,0]),
            "predicted_1": int(matrix[1,1])
        }
    }

    write_metric(cm)

# Ejecutar cálculos
evaluacion("train", y_train, y_train_pred)
evaluacion("test", y_test, y_test_pred)

matriz_confusion("train", y_train, y_train_pred)
matriz_confusion("test", y_test, y_test_pred)

print("🚀 Proceso completado correctamente.")
