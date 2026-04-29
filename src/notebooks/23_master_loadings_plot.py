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
IMAGENES = BASE_DIR / "imagenes"

carreras = ["Matemáticas", "Física"]
sns.set_theme(style="whitegrid")

# Configuración de colores
metric_colors_dict = {'reg': '#1f77b4', 'esf': '#ff7f0e', 'nat': '#2ca02c', 'real': '#d62728'}
metric_labels = {'reg': 'Regularidad', 'esf': 'Esfuerzo', 'nat': 'P. Natural', 'real': 'P. Real'}
semester_palette = sns.color_palette("rocket", 8)
semester_colors_dict = {f's{i}': semester_palette[i-1] for i in range(1, 9)}

def get_pca_data(carrera):
    df = pd.read_excel(DATOS_ACTIVOS / f"{carrera}.xlsx")
    cols = [col for col in df.columns if any(s in col for s in [f's{i}' for i in range(1, 9)])]
    df_clean = df.dropna(subset=cols)
    X_scaled = StandardScaler().fit_transform(df_clean[cols])
    pca = PCA(n_components=5)
    pca.fit(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(5)], index=cols)
    return loadings

# Cargar datos
loadings_mat = get_pca_data("Matemáticas")
loadings_fis = get_pca_data("Física")

def create_master_plot(mode='metric'):
    fig, axes = plt.subplots(5, 2, figsize=(18, 25), sharey=True)
    
    for pc_num in range(1, 6):
        pc_name = f'PC{pc_num}'
        
        for col_idx, (carrera_name, loadings_df) in enumerate([("Matemáticas", loadings_mat), ("Física", loadings_fis)]):
            ax = axes[pc_num-1, col_idx]
            top_loadings = loadings_df[pc_name].sort_values(ascending=False)
            
            if mode == 'metric':
                colors = [metric_colors_dict[var.split('_')[1]] for var in top_loadings.index]
                sns.barplot(x=top_loadings.index, y=top_loadings.values, palette=colors, ax=ax)
            else:
                colors = [semester_colors_dict[var.split('_')[0]] for var in top_loadings.index]
                sns.barplot(x=top_loadings.index, y=top_loadings.values, palette=colors, ax=ax)
            
            ax.set_title(f"{pc_name} - {carrera_name}", fontsize=14)
            ax.set_xticklabels(top_loadings.index, rotation=90, fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("Loading" if col_idx == 0 else "")

    # Leyendas
    if mode == 'metric':
        patches = [mpatches.Patch(color=color, label=label) for key, color, label in 
                   zip(metric_colors_dict.keys(), metric_colors_dict.values(), metric_labels.values())]
        fig.legend(handles=patches, title="Métrica", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)
    else:
        sem_patches = [mpatches.Patch(color=semester_palette[i], label=f"S{i+1}") for i in range(8)]
        fig.legend(handles=sem_patches, title="Semestre", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=8, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(IMAGENES / f"pca_loadings_master_{mode}.png", dpi=200, bbox_inches='tight')
    plt.close()

# Ejecutar
create_master_plot('metric')
create_master_plot('semester')
print("Master plots creados: pca_loadings_master_metric.png y pca_loadings_master_semester.png")
