from PIL import Image
import os
from pathlib import Path

IMAGENES_DIR = Path('/Users/sleepydog/Documents/estadistica/imagenes')

def combinar_horizontal(file1, file2, output_name):
    path1 = IMAGENES_DIR / file1
    path2 = IMAGENES_DIR / file2
    
    if not path1.exists() or not path2.exists():
        print(f"Saltando: alguno de los archivos no existe ({file1}, {file2})")
        return
    
    with Image.open(path1) as img1, Image.open(path2) as img2:
        # Aseguramos que tengan la misma altura redimensionando la segunda
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        new_w2 = int((h1 / h2) * w2)
        img2_res = img2.resize((new_w2, h1), Image.Resampling.LANCZOS)
        
        # Crear imagen nueva
        combined = Image.new('RGB', (w1 + new_w2, h1), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2_res, (w1, 0))
        
        combined.save(IMAGENES_DIR / output_name, "PNG", optimize=True)
        print(f"Creada imagen combinada: {output_name}")

# Lista de gemelas a combinar
gemelas = [
    ('eda_boxplot_gen_matematicas.png', 'eda_boxplot_gen_fisica.png', 'eda_boxplot_combined.png'),
    ('evolucion_metricas_matematicas.png', 'evolucion_metricas_fisica.png', 'evolucion_metricas_combined.png'),
    ('heatmap_matematicas.png', 'heatmap_fisica.png', 'heatmap_combined.png'),
    ('pca_scree_matematicas.png', 'pca_scree_fisica.png', 'pca_scree_combined.png'),
    ('pca_scatter_matematicas.png', 'pca_scatter_fisica.png', 'pca_scatter_combined.png'),
    ('supervivencia_perfiles_matematicas.png', 'supervivencia_perfiles_fisica.png', 'supervivencia_perfiles_combined.png'),
    ('supervivencia_nat_vs_reg_matematicas.png', 'supervivencia_nat_vs_reg_fisica.png', 'supervivencia_nat_vs_reg_combined.png'),
    ('analisis_dual_supervivencia_series_matematicas.png', 'analisis_dual_supervivencia_series_fisica.png', 'analisis_dual_combined.png')
]

print("Iniciando fusión de imágenes gemelas...")
for f1, f2, out in gemelas:
    combinar_horizontal(f1, f2, out)

print("Fusión finalizada.")
