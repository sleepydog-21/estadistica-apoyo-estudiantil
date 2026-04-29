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
    print(f"Recuperando información vía PCA para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    df['semestre_termino'] = df['semestre_termino'].fillna(20)
    
    # Definir grupos (igual que en el script anterior)
    if "Mat" in carrera:
        bins = [0, 8, 12, 15, 20]
        labels = ['Sin Rezago (<= 8)', 'Rezago Leve (9-12)', 'Rezago Medio (13-15)', 'Rezago Severo (16-20)']
    else: # Física
        bins = [0, 9, 12, 15, 20]
        labels = ['Sin Rezago (<= 9)', 'Rezago Leve (10-12)', 'Rezago Medio (13-15)', 'Rezago Severo (16-20)']
    
    df['grupo'] = pd.cut(df['semestre_termino'], bins=bins, labels=labels)
    
    # Filtrar solo alumnos con 8 semestres completos
    target_col = 'semestre_termino'
    cols_s1_s8 = [col for col in df.columns if any(s in col for s in [f's{i}' for i in range(1, 9)])]
    df_clean = df.dropna(subset=cols_s1_s8)
    
    X = df_clean[cols_s1_s8]
    y_grupo = df_clean['grupo']
    y_cont = df_clean[target_col]
    
    # 1. Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Aplicar PCA
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    
    df_results = pd.DataFrame({
        'PC1': scores[:, 0],
        'PC2': scores[:, 1],
        'Grupo': y_grupo,
        'Semestre Término': y_cont
    })
    
    # --- Gráfica: Boxplot de PC1 por Grupo ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x='Grupo', y='PC1', palette="viridis")
    plt.title(f"Capacidad Discriminatoria del PC1 - {carrera}\n(Recuperación de Información del Desempeño S1-S8)", fontsize=14)
    plt.ylabel("Puntaje en PC1 (Componente Principal de Rendimiento)")
    plt.xlabel("Grupo de Graduación")
    
    # Guardar
    nombre_img = f"pca_boxplot_recuperacion_{'matematicas' if 'Mat' in carrera else 'fisica'}.png"
    plt.savefig(IMAGENES / nombre_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Gráfica 2: Correlación Continua ---
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_results, x='PC1', y='Semestre Término', scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.title(f"Correlación PC1 vs Semestre de Término - {carrera}", fontsize=14)
    plt.xlabel("PC1 (Resumen de Desempeño S1-S8)")
    plt.ylabel("Semestre de Término (Activo)")
    
    nombre_img_corr = f"pca_correlacion_target_{'matematicas' if 'Mat' in carrera else 'fisica'}.png"
    plt.savefig(IMAGENES / nombre_img_corr, dpi=300, bbox_inches='tight')
    plt.close()

print("Análisis de recuperación vía PCA finalizado.")
