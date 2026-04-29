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

plt.figure(figsize=(14, 8))

for i, carrera in enumerate(carreras):
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    
    # Caso 1: Todos (imputando 20)
    df_all = df.copy()
    df_all['semestre_termino'] = df_all['semestre_termino'].fillna(20)
    media_all = df_all.groupby('Generación')['semestre_termino'].mean()
    
    # Caso 2: Solo Titulados (excluyendo los que no han terminado)
    # Filtramos los que tienen NaN o 20 (si es que ya había 20s reales, aunque usualmente son censura)
    df_tit = df.dropna(subset=['semestre_termino'])
    df_tit = df_tit[df_tit['semestre_termino'] < 20]
    media_tit = df_tit.groupby('Generación')['semestre_termino'].mean()
    
    # Plotting
    color = 'blue' if i == 0 else 'green'
    plt.plot(media_all.index, media_all.values, marker='o', label=f"{carrera} (Todos, censura 20)", color=color, linestyle='-')
    plt.plot(media_tit.index, media_tit.values, marker='s', label=f"{carrera} (Solo Titulados)", color=color, linestyle='--', alpha=0.6)

plt.title("Impacto de la Censura en la Media de Graduación por Generación", fontsize=16)
plt.xlabel("Generación", fontsize=12)
plt.ylabel("Media Semestre de Término", fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Guardar
plt.savefig(IMAGENES / "comparativa_censura_media.png", dpi=300, bbox_inches='tight')
plt.close()

print("Gráfica comparativa de censura generada.")
