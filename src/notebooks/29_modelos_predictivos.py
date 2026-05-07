import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path('/Users/sleepydog/Documents/tesis estadistica')
DATOS_DIR = BASE_DIR / "Datos" / "Datos_limpios_multiclase"
IMG_DIR = BASE_DIR / "imagenes"

def load_data():
    df_mat = pd.read_excel(DATOS_DIR / "Matemáticas.xlsx")
    df_mat['Carrera'] = 'Matemáticas'
    df_fis = pd.read_excel(DATOS_DIR / "Física.xlsx")
    df_fis['Carrera'] = 'Física'
    return pd.concat([df_mat, df_fis])

df_all = load_data()

# Filtramos filas sin estado final (si las hay)
df_all = df_all.dropna(subset=['estado_final'])

# Crear variables de interacción temprana
df_all['s1_esf_x_real'] = df_all['s1_esf'] * df_all['s1_real']

# Seleccionar predictores de alerta temprana (S1 a S4 unicamente)
features = [
    's1_reg', 's2_reg', 's3_reg', 's4_reg',
    's1_esf', 's2_esf', 's3_esf', 's4_esf',
    's1_nat', 's2_nat', 's3_nat', 's4_nat',
    's1_real', 's2_real', 's3_real', 's4_real',
    's1_esf_x_real'
]

# Remover nulos para el entrenamiento
df_model = df_all.dropna(subset=features + ['estado_final'])

X = df_model[features]
y = df_model['estado_final']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Guardar metricas
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title('Matriz de Confusión: Predicción de Estado Final (S1-S4)')
plt.ylabel('Verdadero')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig(IMG_DIR / 'confusion_matrix.png', dpi=300)
plt.close()

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10
plt.figure(figsize=(10, 6))
plt.title("Top 10 Predictores Tempranos (Importancia Random Forest)")
plt.bar(range(top_n), importances[indices][:top_n], align="center")
plt.xticks(range(top_n), np.array(features)[indices][:top_n], rotation=45)
plt.xlim([-1, top_n])
plt.tight_layout()
plt.savefig(IMG_DIR / 'feature_importance.png', dpi=300)
plt.close()

print("Proceso Finalizado. Gráficas generadas.")
