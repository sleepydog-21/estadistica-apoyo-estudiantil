# %%
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

# Rutas relativas al directorio raíz del proyecto
BASE_DIR = Path("../../").resolve()
DATOS_CRUDOS = BASE_DIR / "Datos" / "Datos_crudos"
DATOS_LIMPIOS = BASE_DIR / "Datos" / "Datos_limpios_truncados"
DATOS_LIMPIOS.mkdir(parents=True, exist_ok=True)

def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

carreras = ["Matematicas", "Fisica"]

def filtrar_bd_bt(df):
    df = df.copy()
    for col in ["BD", "BT"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "BD" in df.columns:
        df = df[df["BD"] != "sí"]
    if "BT" in df.columns:
        df = df[df["BT"] != "sí"]
    return df

def encontrar_col_generacion(df):
    candidatos = ["Generación", "Generacion", "generación", "generacion"]
    for c in candidatos:
        if c in df.columns:
            return c
    return None

def recortar_dataframe(df, n_semestres=8, incluir_semestre_termino=False, incluir_generacion=False):
    df = df.copy()
    sem_cols = [f"s{i}" for i in range(1, n_semestres + 1)]
    cols = ["Cuenta"]

    if incluir_generacion:
        col_gen = encontrar_col_generacion(df)
        if col_gen is not None:
            cols.append(col_gen)

    if incluir_semestre_termino:
        cols.append("semestre_termino")

    cols += sem_cols
    cols = [c for c in cols if c in df.columns]
    return df[cols]

def renombrar_semestres(df, sufijo, n_semestres=8):
    df = df.copy()
    ren = {f"s{i}": f"s{i}_{sufijo}" for i in range(1, n_semestres + 1) if f"s{i}" in df.columns}
    return df.rename(columns=ren)

# %%
archivos_crudos = list(DATOS_CRUDOS.glob("*.xlsx"))

for carrera in carreras:
    print(f"Procesando: {carrera}")
    archivo = None
    for f in archivos_crudos:
        if carrera.lower() == "matematicas" and "aplicadas" in normalize_str(f.name):
            continue
        if carrera.lower() in normalize_str(f.name):
            archivo = f
            break
            
    if not archivo:
        continue
        
    dfs = pd.read_excel(archivo, sheet_name=None)
    df_reg  = filtrar_bd_bt(dfs["Regularidad gregoriana"].copy())
    df_esf  = filtrar_bd_bt(dfs["Índice de esfuerzo"].copy())
    df_nat  = filtrar_bd_bt(dfs["Promedio natural"].copy())
    df_real = filtrar_bd_bt(dfs["Promedio real"].copy())

    sem_cols_20 = [f"s{i}" for i in range(1, 21)]
    sem_cols_reg = [c for c in sem_cols_20 if c in df_reg.columns]
    df_reg[sem_cols_reg] = df_reg[sem_cols_reg].apply(pd.to_numeric, errors="coerce")

    cols_desde_s8 = [f"s{i}" for i in range(8, 21) if f"s{i}" in df_reg.columns]
    mask = df_reg[cols_desde_s8].eq(1)
    tiene_1 = mask.any(axis=1)
    
    primer_col = mask.idxmax(axis=1)
    primer_sem = primer_col.astype(str).str.replace("s", "", regex=False)
    primer_sem = pd.to_numeric(primer_sem, errors='coerce')
    
    # ENFOQUE A CIEGAS: Si no tiene semestre de término, asignar NaN.
    df_reg["semestre_termino"] = np.where(tiene_1, primer_sem, np.nan)

    df_reg_final = recortar_dataframe(df_reg, 8, True, True)
    df_esf_final = recortar_dataframe(df_esf, 8)
    df_nat_final = recortar_dataframe(df_nat, 8)
    df_real_final = recortar_dataframe(df_real, 8)

    df_final = (
        renombrar_semestres(df_reg_final, "reg", 8)
        .merge(renombrar_semestres(df_esf_final, "esf", 8), on="Cuenta", how="inner")
        .merge(renombrar_semestres(df_nat_final, "nat", 8), on="Cuenta", how="inner")
        .merge(renombrar_semestres(df_real_final, "real", 8), on="Cuenta", how="inner")
    )

    # TRUNCAMIENTO: Conservar solo Generaciones <= 2016 y eliminar NaN en semestre_termino
    if 'Generación' in df_final.columns:
        df_final['Generación'] = pd.to_numeric(df_final['Generación'], errors='coerce')
        df_final = df_final[df_final['Generación'] <= 2016]
        
    df_final = df_final.dropna(subset=['semestre_termino'])

    nombre_salida = "Matemáticas" if carrera == "Matematicas" else "Física"
    df_final.to_excel(DATOS_LIMPIOS / f"{nombre_salida}.xlsx", index=False)
    print(f"  -> {nombre_salida} truncado guardado con {len(df_final)} alumnos.")

