# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_LIMPIOS = BASE_DIR / "Datos" / "Datos_limpios_base"
IMAGENES = BASE_DIR / "imagenes"

carreras = ["Matemáticas", "Física"]
sns.set_theme(style="whitegrid")

# %%
for carrera in carreras:
    print(f"Generando histograma de población para: {carrera}")
    archivo = DATOS_LIMPIOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    df = df.dropna(subset=['Generación'])
    df['Generación'] = df['Generación'].astype(int)
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Generación', palette="Blues_d")
    plt.title(f"Población Estudiantil por Generación - {carrera}\n(Excluyendo bajas definitivas y temporales)", fontsize=14)
    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Número de Estudiantes", fontsize=12)
    
    # Agregar los números arriba de cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGENES / f"poblacion_gen_{carrera.lower().replace('á', 'a').replace('í', 'i')}.png", dpi=300)
    plt.close()

print("Generación de gráficas poblacionales terminada.")
