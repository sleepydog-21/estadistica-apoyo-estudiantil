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

for carrera in carreras:
    print(f"Analizando predictores tempranos (S4) para: {carrera}")
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    df['semestre_termino'] = df['semestre_termino'].fillna(20)
    
    # Definir variable binaria: Rezago Crítico (> 12 semestres)
    df['rezago_critico'] = (df['semestre_termino'] > 12).map({True: 'Rezago > 12', False: 'Termina <= 12'})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Regularidad al S4
    sns.boxplot(data=df, x='rezago_critico', y='s4_reg', palette='muted', ax=axes[0])
    axes[0].set_title(f"Regularidad al Semestre 4 vs Resultado Final\n({carrera})", fontsize=14)
    axes[0].set_ylabel("Regularidad (s4_reg)")
    axes[0].set_xlabel("Estado a los 12 Semestres")
    
    # Plot 2: Promedio Real al S4
    sns.boxplot(data=df, x='rezago_critico', y='s4_real', palette='muted', ax=axes[1])
    axes[1].set_title(f"Promedio Real al Semestre 4 vs Resultado Final\n({carrera})", fontsize=14)
    axes[1].set_ylabel("Promedio Real (s4_real)")
    axes[1].set_xlabel("Estado a los 12 Semestres")
    
    plt.suptitle(f"Capacidad Predictiva Temprana (Mitad de Carrera) - {carrera}", fontsize=18, y=1.05)
    plt.tight_layout()
    
    # Guardar
    nombre_img = f"prediccion_temprana_s4_{carrera.lower().replace('á', 'a')}.png"
    plt.savefig(IMAGENES / nombre_img, dpi=300, bbox_inches='tight')
    plt.close()

print("Gráficas de predicción temprana generadas.")
