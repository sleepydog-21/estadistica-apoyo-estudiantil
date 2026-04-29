import pandas as pd
import numpy as np
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_ACTIVOS = BASE_DIR / "Datos" / "Datos_limpios_activos"

carreras = ["Matemáticas", "Física"]
target = "semestre_termino"

results = {}

for carrera in carreras:
    archivo = DATOS_ACTIVOS / f"{carrera}.xlsx"
    if not archivo.exists():
        continue
        
    df = pd.read_excel(archivo)
    df[target] = df[target].fillna(20)
    
    # Seleccionar solo variables s1-s8
    cols_s1_s8 = [col for col in df.columns if any(s in col for s in [f's{i}' for i in range(1, 9)])]
    
    # Calcular correlaciones con el target
    corrs = df[cols_s1_s8 + [target]].corr()[target].drop(target).sort_values(ascending=True) # Ascendente porque buscamos las más negativas (mejor desempeño = menor semestre)
    results[carrera] = corrs

# Crear tabla comparativa de los Top 10 proyectores más fuertes
# (Tomamos las correlaciones más negativas, que indican que a mayor métrica, menor semestre de término)

top_n = 10
table_data = []

for i in range(top_n):
    row = []
    # Matemáticas
    var_m = results["Matemáticas"].index[i]
    val_m = results["Matemáticas"].iloc[i]
    row.extend([var_m.replace('_', '\\_'), f"{val_m:.3f}"])
    
    # Física
    var_f = results["Física"].index[i]
    val_f = results["Física"].iloc[i]
    row.extend([var_f.replace('_', '\\_'), f"{val_f:.3f}"])
    
    table_data.append(row)

# Generar código LaTeX
latex_table = """
\\begin{table}[H]
\\centering
\\footnotesize
\\begin{tabular}{|c|l|c|c|l|c|}
\\hline
\\multicolumn{3}{|c|}{\\textbf{Matemáticas}} & \\multicolumn{3}{c|}{\\textbf{Física}} \\\\ \\hline
\\textbf{Rango} & \\textbf{Variable} & \\textbf{Correlación} & \\textbf{Rango} & \\textbf{Variable} & \\textbf{Correlación} \\\\ \\hline
"""

for i, row in enumerate(table_data):
    latex_table += f"{i+1} & {row[0]} & {row[1]} & {i+1} & {row[2]} & {row[3]} \\\\ \\hline\n"

latex_table += """
\\end{tabular}
\\caption{Top 10 variables con mayor correlación negativa respecto al semestre de término. Valores cercanos a -1 indican que un mejor desempeño en dicha variable predice una graduación más temprana.}
\\label{tab:correlaciones_target}
\\end{table}
"""

print(latex_table)

# Guardar en un archivo temporal por si se necesita
with open(BASE_DIR / "escrito" / "tabla_correlaciones.tex", "w") as f:
    f.write(latex_table)
