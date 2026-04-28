import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_CRUDOS = BASE_DIR / "Datos" / "Datos_crudos"
IMAGENES = BASE_DIR / "imagenes"

def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

carreras = ["Matematicas", "Fisica"]
sns.set_theme(style="whitegrid")

archivos_crudos = list(DATOS_CRUDOS.glob("*.xlsx"))
sem_cols_20 = [f"s{i}" for i in range(1, 21)]

# Archivo de reporte
report_path = BASE_DIR / "escrito" / "stats_correccion.txt"
with open(report_path, "w") as f:
    f.write("Estadísticas de Error - Corrección Fija vs Dinámica\n")
    f.write("="*50 + "\n\n")

for carrera in carreras:
    archivo = None
    for f in archivos_crudos:
        if carrera.lower() == "matematicas" and "aplicadas" in normalize_str(f.name): continue
        if carrera.lower() in normalize_str(f.name):
            archivo = f
            break
    if not archivo: continue
        
    dfs = pd.read_excel(archivo, sheet_name=None)
    df_reg  = dfs["Regularidad gregoriana"].copy()
    df_esf  = dfs["Índice de esfuerzo"].copy()
    
    # Sincronizar cuentas
    df_reg = df_reg.drop_duplicates(subset=['Cuenta'], keep='first').set_index('Cuenta')
    df_esf = df_esf.drop_duplicates(subset=['Cuenta'], keep='first').set_index('Cuenta')
    cuentas_comunes = list(set(df_reg.index) & set(df_esf.index))
    
    # Numérico
    valid_cols = [c for c in sem_cols_20 if c in df_reg.columns]
    df_reg[valid_cols] = df_reg[valid_cols].apply(pd.to_numeric, errors="coerce")
    df_esf[valid_cols] = df_esf[valid_cols].apply(pd.to_numeric, errors="coerce")
    
    # 1. Identificar Semestre Término Cronológico
    cols_desde_s8 = [c for c in valid_cols if int(c[1:]) >= 8]
    mask = df_reg.loc[cuentas_comunes, cols_desde_s8].eq(1)
    tiene_1 = mask.any(axis=1)
    primer_sem = pd.to_numeric(mask.idxmax(axis=1).astype(str).str.replace("s", "", regex=False), errors="coerce")
    df_reg.loc[cuentas_comunes, "semestre_termino_chrono"] = np.where(tiene_1, primer_sem, np.nan)
    
    # Filtrar solo BT == "sí" y sin BD
    if 'BD' in df_reg.columns:
        df_reg['BD'] = df_reg['BD'].astype(str).str.strip().str.lower()
        df_reg = df_reg[df_reg['BD'] != 'sí']
        
    if 'BT' in df_reg.columns:
        df_reg['BT_clean'] = df_reg['BT'].astype(str).str.strip().str.lower()
        df_bt = df_reg[df_reg['BT_clean'] == 'sí'].copy()
        
        # Quedarnos solo con los BT que sí terminaron
        df_bt_egresados = df_bt[df_bt['semestre_termino_chrono'].notna()].copy()
        
        if len(df_bt_egresados) == 0: continue
            
        # Contar semestres inactivos antes de su semestre de egreso
        inactivos_list = []
        for cuenta in df_bt_egresados.index:
            s_term = int(df_bt_egresados.loc[cuenta, 'semestre_termino_chrono'])
            esf_vals = df_esf.loc[cuenta, [f"s{i}" for i in range(1, s_term + 1)]].values.astype(float)
            inactivos = np.isnan(esf_vals).sum()
            inactivos_list.append(inactivos)
            
        df_bt_egresados['semestres_inactivos'] = inactivos_list
        
        # Cálculos de Corrección
        C = df_bt_egresados['semestres_inactivos'].mean()
        df_bt_egresados['semestre_termino_dinamico'] = df_bt_egresados['semestre_termino_chrono'] - df_bt_egresados['semestres_inactivos']
        df_bt_egresados['semestre_termino_fijo'] = df_bt_egresados['semestre_termino_chrono'] - C
        
        # Métricas de error
        error_absoluto = np.abs(df_bt_egresados['semestre_termino_fijo'] - df_bt_egresados['semestre_termino_dinamico'])
        mae = error_absoluto.mean()
        max_error = error_absoluto.max()
        
        # Reportar
        carrera_nombre = "Matemáticas" if carrera == "Matematicas" else "Física"
        with open(report_path, "a") as f:
            f.write(f"CARRERA: {carrera_nombre}\n")
            f.write(f" - Egresados con BT: {len(df_bt_egresados)}\n")
            f.write(f" - Promedio de semestres inactivos (C): {C:.2f}\n")
            f.write(f" - MAE de corrección fija vs dinámica: {mae:.2f} semestres\n")
            f.write(f" - Error Máximo: {max_error:.2f} semestres\n\n")
            
        # Graficar caso 2016 de BT egresados
        df_bt_egresados['Generación'] = pd.to_numeric(df_bt_egresados['Generación'], errors='coerce')
        df_2016 = df_bt_egresados[df_bt_egresados['Generación'] == 2016]
        
        if len(df_2016) > 0:
            df_plot = df_2016[['semestre_termino_chrono', 'semestre_termino_dinamico']].melt(
                var_name='Tipo de Cálculo', value_name='Semestre de Titulación'
            )
            df_plot['Tipo de Cálculo'] = df_plot['Tipo de Cálculo'].map({
                'semestre_termino_chrono': 'Cronológico (Con pausa BT)',
                'semestre_termino_dinamico': 'Activo (Corrección Dinámica)'
            })
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_plot, x='Semestre de Titulación', hue='Tipo de Cálculo', 
                         multiple='dodge', binwidth=1, palette='Set1')
            plt.title(f"Impacto de la Corrección Dinámica en Alumnos con BT\n(Generación 2016 - {carrera_nombre})", fontsize=14)
            plt.xlabel("Semestre en el que se alcanzan requisitos de titulación")
            plt.ylabel("Frecuencia")
            plt.savefig(IMAGENES / f"correccion_bt_2016_{carrera.lower()}.png", dpi=300)
            plt.close()

print("Estadísticas calculadas y guardadas.")
