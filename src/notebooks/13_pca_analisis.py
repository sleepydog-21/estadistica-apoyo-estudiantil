import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / "Datos" / "Datos_limpios_activos"
IMAGENES = BASE_DIR / "imagenes"

carreras = ["Matemáticas", "Física"]

sns.set_theme(style="whitegrid")

for carrera in carreras:
    print(f"Iniciando PCA para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    
    if not archivo.exists():
        print(f"Archivo no encontrado: {archivo}")
        continue
        
    df = pd.read_excel(archivo)
    
    # Target: Imputar 20 a los no titulados
    target_col = 'semestre_termino'
    y = df[target_col].fillna(20).values
    
    # Predictores: Excluir metadata y el target
    metadata_cols = ['Cuenta', 'Generación', target_col]
    X_raw = df.drop(columns=metadata_cols, errors='ignore')
    
    # 1. Imputación (PCA no acepta NaNs)
    # Usamos la media para no sesgar drásticamente la distribución
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    
    # 2. Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Aplicar PCA
    pca = PCA(n_components=10) 
    X_pca = pca.fit_transform(X_scaled)
    
    # --- Gráfica 1: Varianza Explicada ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title(f"Varianza Acumulada Explicada - {carrera}")
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza Explicada")
    plt.grid(True)
    plt.savefig(IMAGENES / f"pca_varianza_{carrera.lower().replace('á', 'a')}.png", dpi=300)
    plt.close()
    
    # --- Gráfica 2: Proyección PC1 vs PC2 coloreada por Semestre de Término ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis_r', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Semestre de Término')
    plt.title(f"PCA: PC1 vs PC2 - {carrera}\n(Coloreado por Variable Objetivo)", fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.savefig(IMAGENES / f"pca_scatter_{carrera.lower().replace('á', 'a')}.png", dpi=300)
    plt.close()
    
    # --- Gráfica 3: Loadings (Contribución de variables a los primeros 2 componentes) ---
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(10)], 
        index=X_raw.columns
    )
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(loadings.iloc[:, :3], annot=False, cmap='RdBu_r', center=0)
    plt.title(f"Loadings de PCA (PC1, PC2, PC3) - {carrera}")
    plt.tight_layout()
    plt.savefig(IMAGENES / f"pca_loadings_{carrera.lower().replace('á', 'a')}.png", dpi=300)
    plt.close()

print("Análisis de PCA finalizado.")
