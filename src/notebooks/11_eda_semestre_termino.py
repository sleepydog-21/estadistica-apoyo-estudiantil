import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / 'Datos' / 'Datos_limpios_activos'
IMAGENES = BASE_DIR / 'imagenes'

sns.set_theme(style='whitegrid')

# Cargar ambos datasets
df_mate = pd.read_excel(DATOS_ACTIVOS / 'Matemáticas.xlsx')
df_mate['Carrera'] = 'Matemáticas'

df_fisica = pd.read_excel(DATOS_ACTIVOS / 'Física.xlsx')
df_fisica['Carrera'] = 'Física'

df = pd.concat([df_mate, df_fisica], ignore_index=True)

# Imputar 20 a los no titulados (o censurados en el modelo activo)
df['semestre_termino'] = df['semestre_termino'].fillna(20)
df['Generación'] = pd.to_numeric(df['Generación'], errors='coerce')
df = df.dropna(subset=['Generación'])
df['Generación'] = df['Generación'].astype(int)

# 1. Gráfica de Media por Generación
df_mean = df.groupby(['Generación', 'Carrera'])['semestre_termino'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_mean, x='Generación', y='semestre_termino', hue='Carrera', marker='o', palette='Set1', linewidth=2)
plt.title('Promedio del Semestre de Término Activo por Generación', fontsize=14)
plt.ylabel('Semestre de Término Promedio (Censura en 20)', fontsize=12)
plt.xlabel('Generación', fontsize=12)
plt.xticks(df['Generación'].unique(), rotation=45)
plt.axhline(y=20, color='r', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(IMAGENES / 'eda_media_semestre_termino.png', dpi=300)
plt.show()

# 2. Boxplots por generación (Distribución/Histogramas)
for carrera in ['Matemáticas', 'Física']:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[df['Carrera'] == carrera], x='Generación', y='semestre_termino', color='teal' if carrera == 'Matemáticas' else 'coral')
    plt.title(f'Distribución del Semestre de Término Activo por Generación - {carrera}', fontsize=14)
    plt.ylabel('Semestre de Término', fontsize=12)
    plt.xlabel('Generación', fontsize=12)
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(IMAGENES / f'eda_boxplot_gen_{carrera.lower().replace("á", "a")}.png', dpi=300)
    plt.show()
