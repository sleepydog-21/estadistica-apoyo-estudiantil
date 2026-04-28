# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_CRUDOS = BASE_DIR / 'Datos' / 'Datos_crudos'
IMAGENES = BASE_DIR / 'imagenes'

def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

carreras = ['Matematicas', 'Fisica']
sns.set_theme(style='whitegrid')

archivos_crudos = list(DATOS_CRUDOS.glob('*.xlsx'))

for carrera in carreras:
    archivo = None
    for f in archivos_crudos:
        if carrera.lower() == 'matematicas' and 'aplicadas' in normalize_str(f.name): continue
        if carrera.lower() in normalize_str(f.name):
            archivo = f
            break
            
    if not archivo: continue
        
    df_raw = pd.read_excel(archivo, sheet_name='Regularidad gregoriana')
    
    # 1. Total (Sin filtrar)
    df_raw = df_raw.dropna(subset=['Generación'])
    df_raw['Generación'] = df_raw['Generación'].astype(int)
    count_raw = df_raw.groupby('Generación').size().reset_index(name='Total Inscritos')
    
    # 2. Filtrado (Sin BD/BT)
    df_filtered = df_raw.copy()
    for col in ['BD', 'BT']:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype(str).str.strip().str.lower()
    if 'BD' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['BD'] != 'sí']
    if 'BT' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['BT'] != 'sí']
        
    count_filtered = df_filtered.groupby('Generación').size().reset_index(name='Población Activa (Sin BD/BT)')
    
    # Merge
    df_plot = pd.merge(count_raw, count_filtered, on='Generación', how='outer').fillna(0)
    df_melt = pd.melt(df_plot, id_vars='Generación', var_name='Estado', value_name='Número de Estudiantes')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melt, x='Generación', y='Número de Estudiantes', hue='Estado', palette='Set2')
    
    carrera_nombre = 'Matemáticas' if carrera == 'Matematicas' else 'Física'
    plt.title(f'Impacto de las Bajas en la Población por Generación - {carrera_nombre}', fontsize=14)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 2),
                        textcoords='offset points')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGENES / f'poblacion_comparativa_{carrera.lower()}.png', dpi=300)
    plt.close()

print("Generación de gráficas comparativas terminada.")
