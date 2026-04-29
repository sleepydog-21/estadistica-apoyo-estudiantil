import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / "Datos" / "Datos_limpios_activos"
IMAGENES = BASE_DIR / "imagenes"

carreras = ["Matemáticas", "Física"]
sns.set_theme(style="whitegrid")

for carrera in carreras:
    print(f"Iniciando PCA detallado para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    
    # Filtrar solo alumnos con 8 semestres completos
    cols_s1_s8 = [col for col in df.columns if any(s in col for s in [f's{i}' for i in range(1, 9)])]
    df_clean = df.dropna(subset=cols_s1_s8)
    X = df_clean[cols_s1_s8]
    
    # 1. Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA Completo (hasta 32 componentes)
    pca = PCA()
    pca.fit(X_scaled)
    
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # --- Gráfica 1: Scree Plot (Elbow) ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, align='center', label='Varianza Individual', color='skyblue')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Varianza Acumulada', color='navy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Varianza')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Varianza')
    
    plt.title(f"Scree Plot - Varianza Explicada ({carrera})", fontsize=15)
    plt.xlabel("Número de Componentes Principales")
    plt.ylabel("Ratio de Varianza Explicada")
    plt.xticks(range(1, 11)) # Mostrar solo los primeros 10 para mayor claridad
    plt.xlim(0.5, 10.5)
    plt.legend(loc='best')
    
    plt.savefig(IMAGENES / f"pca_scree_{carrera.lower().replace('á', 'a')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Gráfica 2: Comparación de Loadings (PC1 y PC2) ---
    # Vamos a tomar los loadings de los 2 primeros componentes
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(cols_s1_s8))],
        index=cols_s1_s8
    )
    
    # Graficamos PC1 y PC2
    for pc in ['PC1', 'PC2']:
        plt.figure(figsize=(14, 6))
        top_loadings = loadings[pc].sort_values(ascending=False)
        
        # Color diferenciado por métrica
        colors = []
        for var in top_loadings.index:
            if 'reg' in var: colors.append('#1f77b4')
            elif 'esf' in var: colors.append('#ff7f0e')
            elif 'nat' in var: colors.append('#2ca02c')
            else: colors.append('#d62728') # real
            
        sns.barplot(x=top_loadings.index, y=top_loadings.values, palette=colors)
        plt.title(f"Cargas (Loadings) de {pc} - {carrera}", fontsize=15)
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel("Peso del Coeficiente")
        
        # Guardar
        plt.savefig(IMAGENES / f"pca_loadings_{pc.lower()}_{carrera.lower().replace('á', 'a')}.png", dpi=300, bbox_inches='tight')
        plt.close()

print("PCA detallado y gráficas de loadings generadas.")
