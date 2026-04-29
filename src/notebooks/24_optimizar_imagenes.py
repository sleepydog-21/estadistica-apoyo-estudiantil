from PIL import Image
import os
from pathlib import Path

# Configuración
IMAGENES_DIR = Path('/Users/sleepydog/Documents/estadistica/imagenes')

# Lista de imágenes que deben mantenerse en ALTA calidad (300 DPI)
HIGH_QUALITY = [
    'pca_loadings_master_metric.png',
    'pca_loadings_master_semester.png',
    'pca_scatter_matematicas.png',
    'pca_scatter_fisica.png',
    'evolucion_metricas_matematicas.png',
    'evolucion_metricas_fisica.png',
    'heatmap_matematicas.png',
    'heatmap_fisica.png',
    'pca_boxplot_recuperacion_matematicas.png',
    'pca_boxplot_recuperacion_fisica.png'
]

# Extensiones de imagen a procesar
EXTENSIONS = ['.png', '.jpg', '.jpeg']

print("Iniciando optimización de imágenes...")

for img_file in IMAGENES_DIR.iterdir():
    if img_file.suffix.lower() in EXTENSIONS:
        if img_file.name in HIGH_QUALITY:
            print(f"Manteniendo ALTA calidad: {img_file.name}")
            continue # No tocamos las de alta calidad
            
        # Para las demás, reducimos calidad
        try:
            with Image.open(img_file) as img:
                # Si es RGBA, convertimos a RGB para ahorrar espacio si no hay transparencia crítica
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Guardamos con optimización y menor resolución efectiva (DPI)
                original_size = os.path.getsize(img_file)
                img.save(img_file, "PNG", optimize=True)
                new_size = os.path.getsize(img_file)
                
                reduction = (original_size - new_size) / 1024
                print(f"Optimizado: {img_file.name} (Reducción: {reduction:.2f} KB)")
        except Exception as e:
            print(f"Error procesando {img_file.name}: {e}")

print("Optimización finalizada.")
