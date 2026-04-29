from PIL import Image
import os
from pathlib import Path

IMAGENES_DIR = Path('/Users/sleepydog/Documents/estadistica/imagenes')
MAX_WIDTH = 1200 # Ancho máximo suficiente para una tesis

print("Iniciando optimización AGRESIVA de imágenes...")

for img_file in IMAGENES_DIR.iterdir():
    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        try:
            with Image.open(img_file) as img:
                # 1. Convertir a RGB (quitar transparencias para ahorrar espacio)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 2. Redimensionar si es muy grande
                w, h = img.size
                if w > MAX_WIDTH:
                    new_h = int((MAX_WIDTH / w) * h)
                    img = img.resize((MAX_WIDTH, new_h), Image.Resampling.LANCZOS)
                    print(f"Redimensionado: {img_file.name} de {w}px a {MAX_WIDTH}px")
                
                # 3. Guardar con optimización máxima
                original_size = os.path.getsize(img_file)
                # Usamos un poco de compresión para PNG (quantization) si es necesario
                # Pero primero probamos con optimización estándar
                img.save(img_file, "PNG", optimize=True)
                new_size = os.path.getsize(img_file)
                
                reduction = (original_size - new_size) / 1024
                if reduction > 0:
                    print(f"Comprimido: {img_file.name} (-{reduction:.2f} KB)")
                else:
                    print(f"Ya estaba optimizado: {img_file.name}")
                    
        except Exception as e:
            print(f"Error procesando {img_file.name}: {e}")

print("Optimización agresiva finalizada.")
