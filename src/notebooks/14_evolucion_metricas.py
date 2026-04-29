import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / "Datos" / "Datos_limpios_activos"
IMAGENES = BASE_DIR / "imagenes"

carreras = ["Matemáticas", "Física"]
metrics = ["reg", "esf", "nat", "real"]
metric_names = {
    "reg": "Regularidad",
    "esf": "Índice de Esfuerzo",
    "nat": "Promedio Natural",
    "real": "Promedio Real"
}

sns.set_theme(style="whitegrid")

for carrera in carreras:
    print(f"Analizando evolución de métricas para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    df['semestre_termino'] = df['semestre_termino'].fillna(20)
    
    # Definir grupos por semestre de término específicos por carrera
    if "Mat" in carrera:
        bins = [0, 8, 12, 15, 20]
        labels = ['Ideal (<= 8)', 'Leve (9-12)', 'Medio (13-15)', 'Severo/Censura (16-20)']
    else: # Física
        bins = [0, 9, 12, 15, 20]
        labels = ['Ideal (<= 9)', 'Leve (10-12)', 'Medio (13-15)', 'Severo/Censura (16-20)']
        
    df['grupo'] = pd.cut(df['semestre_termino'], bins=bins, labels=labels)
    
    # Filtrar solo alumnos con 8 semestres completos para que la línea sea continua
    # (Siguiendo la lógica del PCA anterior)
    cols_s1_s8 = [f's{i}_{m}' for i in range(1, 9) for m in metrics]
    df_clean = df.dropna(subset=cols_s1_s8)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    palette = sns.color_palette("viridis", n_colors=4)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Preparar datos para el plot
        cols = [f's{j}_{metric}' for j in range(1, 9)]
        df_metric = df_clean.groupby('grupo', observed=True)[cols].mean().reset_index()
        
        # Melt para seaborn
        df_melt = df_metric.melt(id_vars='grupo', var_name='Semestre', value_name='Valor')
        df_melt['Semestre'] = df_melt['Semestre'].str.extract('(\d+)').astype(int)
        
        sns.lineplot(data=df_melt, x='Semestre', y='Valor', hue='grupo', marker='o', ax=ax, palette=palette, linewidth=2.5)
        
        ax.set_title(f"Evolución de {metric_names[metric]} (S1-S8)", fontsize=14)
        ax.set_xlabel("Semestre")
        ax.set_ylabel("Valor Promedio")
        ax.set_ylim(0, 10.5 if 'promedio' in metric_names[metric].lower() else 1.1)
        ax.legend(title="Grupo de Graduación", fontsize='small')
        
    plt.suptitle(f"Divergencia Académica por Tiempo de Egreso - {carrera}", fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Guardar
    nombre_img = f"evolucion_metricas_{'matematicas' if 'Mat' in carrera else 'fisica'}.png"
    plt.savefig(IMAGENES / nombre_img, dpi=300, bbox_inches='tight')
    plt.close()

print("Gráficas de evolución generadas.")
