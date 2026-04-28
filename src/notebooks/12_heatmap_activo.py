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

sns.set_theme(style="white")

for carrera in carreras:
    print(f"Generando matriz de correlación para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    
    if not archivo.exists():
        print(f"Archivo no encontrado: {archivo}")
        continue
        
    df = pd.read_excel(archivo)
    
    # Imputar 20 a los no titulados para la correlación (censura)
    df['semestre_termino'] = df['semestre_termino'].fillna(20)
    
    # Seleccionar solo columnas numéricas y relevantes
    # (Excluimos Cuenta, Generación para el heatmap si se desea, o lo dejamos)
    cols_to_drop = ['Cuenta']
    if 'Plan' in df.columns: cols_to_drop.append('Plan')
    
    df_num = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    # Calcular correlación
    corr = df_num.corr()
    
    # Graficar
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f"Matriz de Correlación - {carrera} (Tiempo Activo)", fontsize=16)
    plt.tight_layout()
    
    # Guardar
    nombre_img = f"heatmap_{carrera.lower().replace('á', 'a')}.png"
    plt.savefig(IMAGENES / nombre_img, dpi=300)
    plt.close()

print("Heatmaps actualizados.")
