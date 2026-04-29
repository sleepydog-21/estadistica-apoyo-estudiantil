import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / "Datos" / "Datos_limpios_activos"
IMAGENES = BASE_DIR / "imagenes" / "pca_loadings_detalle"
IMAGENES.mkdir(parents=True, exist_ok=True)

carreras = ["Matemáticas", "Física"]
sns.set_theme(style="whitegrid")

# Diccionarios de colores
metric_colors_dict = {
    'reg': '#1f77b4',  # Azul
    'esf': '#ff7f0e',  # Naranja
    'nat': '#2ca02c',  # Verde
    'real': '#d62728'  # Rojo
}
metric_labels = {
    'reg': 'Regularidad',
    'esf': 'Índice de Esfuerzo',
    'nat': 'Promedio Natural',
    'real': 'Promedio Real'
}

# Paleta para semestres (S1-S8)
semester_palette = sns.color_palette("rocket", 8)
semester_colors_dict = {f's{i}': semester_palette[i-1] for i in range(1, 9)}

for carrera in carreras:
    print(f"Generando loadings detallados para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    cols_s1_s8 = [col for col in df.columns if any(s in col for s in [f's{i}' for i in range(1, 9)])]
    df_clean = df.dropna(subset=cols_s1_s8)
    X = df_clean[cols_s1_s8]
    
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=5)
    pca.fit(X_scaled)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(5)],
        index=cols_s1_s8
    )
    
    for pc_num in range(1, 6):
        pc_name = f'PC{pc_num}'
        
        # --- Gráfica Tipo 1: Por Métrica ---
        plt.figure(figsize=(15, 7))
        top_loadings = loadings[pc_name].sort_values(ascending=False)
        
        colors_m = [metric_colors_dict[var.split('_')[1]] for var in top_loadings.index]
        sns.barplot(x=top_loadings.index, y=top_loadings.values, palette=colors_m)
        
        # Leyenda manual para métricas
        patches = [mpatches.Patch(color=color, label=label) for key, color, label in 
                   zip(metric_colors_dict.keys(), metric_colors_dict.values(), metric_labels.values())]
        plt.legend(handles=patches, title="Métrica", loc='upper right', fontsize='small')
        
        plt.title(f"Loadings {pc_name} ({carrera}) - Coloreado por Métrica", fontsize=15)
        plt.xticks(rotation=90)
        plt.savefig(IMAGENES / f"loadings_{pc_name.lower()}_{carrera.lower().replace('á', 'a')}_metrica.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Gráfica Tipo 2: Por Semestre ---
        plt.figure(figsize=(15, 7))
        # (Usamos el mismo orden de variables para facilitar la comparación)
        colors_s = [semester_colors_dict[var.split('_')[0]] for var in top_loadings.index]
        sns.barplot(x=top_loadings.index, y=top_loadings.values, palette=colors_s)
        
        # Leyenda manual para semestres
        sem_patches = [mpatches.Patch(color=semester_palette[i], label=f"Semestre {i+1}") for i in range(8)]
        plt.legend(handles=sem_patches, title="Semestre", loc='upper right', ncol=2, fontsize='x-small')
        
        plt.title(f"Loadings {pc_name} ({carrera}) - Coloreado por Semestre", fontsize=15)
        plt.xticks(rotation=90)
        plt.savefig(IMAGENES / f"loadings_{pc_name.lower()}_{carrera.lower().replace('á', 'a')}_semestre.png", dpi=300, bbox_inches='tight')
        plt.close()

print(f"Se han generado 20 gráficas de loadings en {IMAGENES}")
