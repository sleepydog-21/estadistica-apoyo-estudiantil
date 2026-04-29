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
sns.set_theme(style="whitegrid")

# 1. Preparación de datos agregados
dfs = []
for carrera in carreras:
    df = pd.read_excel(DATOS_ACTIVOS / f"{carrera}.xlsx")
    df['Carrera'] = carrera
    df['semestre_termino_limpio'] = df['semestre_termino'].dropna() # Solo los que terminaron
    
    # Métricas agregadas S1-S8
    df['Esfuerzo_Medio'] = df[[f's{i}_esf' for i in range(1, 9)]].mean(axis=1)
    df['Promedio_Real_Medio'] = df[[f's{i}_real' for i in range(1, 9)]].mean(axis=1)
    
    dfs.append(df)

df_all = pd.concat(dfs)

# --- GRÁFICA 1: Esfuerzo vs Promedio (Correlación con el Target) ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.regplot(data=df_all[df_all['Carrera']=='Matemáticas'], x='Esfuerzo_Medio', y='semestre_termino', 
            scatter_kws={'alpha':0.1, 'color':'blue'}, line_kws={'color':'navy'}, label='Matemáticas')
sns.regplot(data=df_all[df_all['Carrera']=='Física'], x='Esfuerzo_Medio', y='semestre_termino', 
            scatter_kws={'alpha':0.1, 'color':'red'}, line_kws={'color':'darkred'}, label='Física')
plt.title("Esfuerzo (Créditos) vs Tiempo de Egreso", fontsize=14)
plt.legend()

plt.subplot(1, 2, 2)
sns.regplot(data=df_all[df_all['Carrera']=='Matemáticas'], x='Promedio_Real_Medio', y='semestre_termino', 
            scatter_kws={'alpha':0.1, 'color':'blue'}, line_kws={'color':'navy'}, label='Matemáticas')
sns.regplot(data=df_all[df_all['Carrera']=='Física'], x='Promedio_Real_Medio', y='semestre_termino', 
            scatter_kws={'alpha':0.1, 'color':'red'}, line_kws={'color':'darkred'}, label='Física')
plt.title("Calificación vs Tiempo de Egreso", fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig(IMAGENES / "eda_esfuerzo_vs_promedio.png", dpi=300)
plt.close()

# --- GRÁFICA 2: Inercia Académica (Autocorrelación Semestral) ---
plt.figure(figsize=(16, 6))

for i, carrera in enumerate(carreras):
    plt.subplot(1, 2, i+1)
    df_c = df_all[df_all['Carrera'] == carrera]
    # Matriz de correlación entre promedios reales S1 a S8
    cols_real = [f's{i}_real' for i in range(1, 9)]
    corr_matrix = df_c[cols_real].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    plt.title(f"Inercia Académica: Correlación S1-S8 ({carrera})", fontsize=14)

plt.tight_layout()
plt.savefig(IMAGENES / "eda_inercia_semestral.png", dpi=300)
plt.close()

# --- GRÁFICA 3: Comparativa de Densidad Mate vs Física (Sin Censura) ---
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_all.dropna(subset=['semestre_termino']), x='semestre_termino', hue='Carrera', fill=True, common_norm=False, palette='viridis')
plt.axvline(x=9, color='red', linestyle='--', label='Tiempo Ideal (Física)')
plt.axvline(x=8, color='blue', linestyle='--', label='Tiempo Ideal (Mate)')
plt.title("Distribución Real del Tiempo de Egreso (Solo Titulados)", fontsize=15)
plt.xlabel("Semestre de Término")
plt.ylabel("Densidad de Estudiantes")
plt.xlim(7, 21)
plt.legend()

plt.savefig(IMAGENES / "eda_densidad_egreso_comparativa.png", dpi=300)
plt.close()

print("Análisis final del EDA completado y gráficas generadas.")
