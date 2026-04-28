# %%
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

BASE_DIR = Path("../../").resolve()
DATOS_CRUDOS = BASE_DIR / "Datos" / "Datos_crudos"
DATOS_LIMPIOS = BASE_DIR / "Datos" / "Datos_limpios_multiclase"
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

    # Identificar semestre de término real
    cols_desde_s8 = [f"s{i}" for i in range(8, 21) if f"s{i}" in df_reg.columns]
    mask = df_reg[cols_desde_s8].eq(1)
    tiene_1 = mask.any(axis=1)
    primer_sem = pd.to_numeric(mask.idxmax(axis=1).astype(str).str.replace("s", "", regex=False), errors='coerce')

    df_reg["semestre_termino"] = np.where(tiene_1, primer_sem, np.nan)

    # Definir "Activo" o "Abandono" basado en el esfuerzo de los ultimos 4 semestres registrados (s17 a s20)
    # o bien si no hay datos.
    sem_cols_esf = [c for c in sem_cols_20 if c in df_esf.columns]
    df_esf[sem_cols_esf] = df_esf[sem_cols_esf].apply(pd.to_numeric, errors="coerce")
    
    # Calcular esfuerzo promedio en los últimos semestres (s15 a s20) para determinar abandono
    cols_finales_esf = [f"s{i}" for i in range(15, 21) if f"s{i}" in df_esf.columns]
    esfuerzo_final = df_esf[cols_finales_esf].sum(axis=1, skipna=True)
    
    def clasificar_estado(row, esf_final):
        sem = row["semestre_termino"]
        if pd.notna(sem):
            if sem <= 10:
                return "Titulado a tiempo"
            else:
                return "Titulado con rezago"
        else:
            # Si no se tituló y su esfuerzo en los ultimos semestres es 0, asumimos abandono
            if esf_final == 0:
                return "Abandono silencioso"
            else:
                return "Activo / Aún cursando"

    df_reg["estado_final"] = [clasificar_estado(row, esf) for (_, row), esf in zip(df_reg.iterrows(), esfuerzo_final)]

    df_reg_final = recortar_dataframe(df_reg, 8, True, True)
    df_reg_final["estado_final"] = df_reg["estado_final"] # Añadir estado
    
    df_esf_final = recortar_dataframe(df_esf, 8)
    df_nat_final = recortar_dataframe(df_nat, 8)
    df_real_final = recortar_dataframe(df_real, 8)

    df_final = (
        renombrar_semestres(df_reg_final, "reg", 8)
        .merge(renombrar_semestres(df_esf_final, "esf", 8), on="Cuenta", how="inner")
        .merge(renombrar_semestres(df_nat_final, "nat", 8), on="Cuenta", how="inner")
        .merge(renombrar_semestres(df_real_final, "real", 8), on="Cuenta", how="inner")
    )

    nombre_salida = "Matemáticas" if carrera == "Matematicas" else "Física"
    df_final.to_excel(DATOS_LIMPIOS / f"{nombre_salida}.xlsx", index=False)
    print(f"  -> {nombre_salida} multiclase guardado con etiquetas de estado_final.")

