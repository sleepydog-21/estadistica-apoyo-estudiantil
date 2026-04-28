# %%
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

# Rutas relativas al directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_CRUDOS = BASE_DIR / "Datos" / "Datos_crudos"
DATOS_LIMPIOS = BASE_DIR / "Datos" / "Datos_limpios"
DATOS_LIMPIOS.mkdir(parents=True, exist_ok=True)

# Normalizar cadenas para evitar problemas de unicode (acentos)
def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

carreras = [
    "Matematicas", 
    "Fisica"
]

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
# Procesamiento por carrera
archivos_crudos = list(DATOS_CRUDOS.glob("*.xlsx"))

for carrera in carreras:
    print(f"Procesando: {carrera}")
    
    # Buscar el archivo correspondiente ignorando acentos y mayúsculas
    archivo = None
    for f in archivos_crudos:
        # Excluir 'Matemáticas Aplicadas' si estamos buscando solo 'Matemáticas'
        if carrera.lower() == "matematicas" and "aplicadas" in normalize_str(f.name):
            continue
            
        if carrera.lower() in normalize_str(f.name):
            archivo = f
            break
            
    if not archivo:
        print(f"No se encontró archivo para {carrera}")
        continue
        
    dfs = pd.read_excel(archivo, sheet_name=None)

    df_reg  = dfs["Regularidad gregoriana"].copy()
    df_esf  = dfs["Índice de esfuerzo"].copy()
    df_nat  = dfs["Promedio natural"].copy()
    df_real = dfs["Promedio real"].copy()

    df_reg  = filtrar_bd_bt(df_reg)
    df_esf  = filtrar_bd_bt(df_esf)
    df_nat  = filtrar_bd_bt(df_nat)
    df_real = filtrar_bd_bt(df_real)

    sem_cols_20 = [f"s{i}" for i in range(1, 21)]
    sem_cols_reg = [c for c in sem_cols_20 if c in df_reg.columns]
    df_reg[sem_cols_reg] = df_reg[sem_cols_reg].apply(pd.to_numeric, errors="coerce")

    cols_desde_s8 = [f"s{i}" for i in range(8, 21) if f"s{i}" in df_reg.columns]
    mask = df_reg[cols_desde_s8].eq(1)

    tiene_1 = mask.any(axis=1)
    primer_col = mask.idxmax(axis=1)
    primer_sem = primer_col.astype(str).str.replace("s", "", regex=False)
    primer_sem = pd.to_numeric(primer_sem, errors='coerce').fillna(20).astype(int)

    df_reg["semestre_termino"] = np.where(tiene_1, primer_sem, 20)

    df_reg_final = recortar_dataframe(
        df_reg,
        n_semestres=8,
        incluir_semestre_termino=True,
        incluir_generacion=True
    )

    df_esf_final = recortar_dataframe(df_esf, n_semestres=8)
    df_nat_final = recortar_dataframe(df_nat, n_semestres=8)
    df_real_final = recortar_dataframe(df_real, n_semestres=8)

    df_reg_r  = renombrar_semestres(df_reg_final,  "reg",  8)
    df_esf_r  = renombrar_semestres(df_esf_final,  "esf",  8)
    df_nat_r  = renombrar_semestres(df_nat_final,  "nat",  8)
    df_real_r = renombrar_semestres(df_real_final, "real", 8)

    df_final = (
        df_reg_r
        .merge(df_esf_r,  on="Cuenta", how="inner")
        .merge(df_nat_r,  on="Cuenta", how="inner")
        .merge(df_real_r, on="Cuenta", how="inner")
    )

    # Nombres de salida correctos
    nombre_salida = "Matemáticas" if carrera == "Matematicas" else "Física"
    archivo_salida = DATOS_LIMPIOS / f"{nombre_salida}.xlsx"
    df_final.to_excel(archivo_salida, index=False)
    print(f"  -> Guardado en {archivo_salida.name}")

print("Proceso de limpieza completado.")
