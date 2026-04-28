import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_CRUDOS = BASE_DIR / "Datos" / "Datos_crudos"
DATOS_LIMPIOS = BASE_DIR / "Datos" / "Datos_limpios_activos"
DATOS_LIMPIOS.mkdir(parents=True, exist_ok=True)

def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

carreras = ["Matematicas", "Fisica"]

def filtrar_bd(df):
    df = df.copy()
    if "BD" in df.columns:
        df["BD"] = df["BD"].astype(str).str.strip().str.lower()
        df = df[df["BD"] != "sí"]
    return df

def encontrar_col_generacion(df):
    for c in ["Generación", "Generacion", "generación", "generacion"]:
        if c in df.columns: return c
    return None

def renombrar_semestres(df, sufijo, n_semestres=8):
    ren = {f"s{i}": f"s{i}_{sufijo}" for i in range(1, n_semestres + 1) if f"s{i}" in df.columns}
    return df.rename(columns=ren)

# Columnas de semestres
sem_cols_20 = [f"s{i}" for i in range(1, 21)]

for carrera in carreras:
    print(f"Procesando: {carrera} con Tiempo Activo")
    archivo = None
    for f in list(DATOS_CRUDOS.glob("*.xlsx")):
        if carrera.lower() == "matematicas" and "aplicadas" in normalize_str(f.name): continue
        if carrera.lower() in normalize_str(f.name):
            archivo = f
            break
    if not archivo: continue
        
    dfs = pd.read_excel(archivo, sheet_name=None)
    df_reg  = filtrar_bd(dfs["Regularidad gregoriana"].copy())
    df_esf  = dfs["Índice de esfuerzo"].copy()
    df_nat  = dfs["Promedio natural"].copy()
    df_real = dfs["Promedio real"].copy()
    
    # Sincronizar cuentas y eliminar duplicados
    for df in [df_reg, df_esf, df_nat, df_real]:
        df.drop_duplicates(subset=['Cuenta'], keep='first', inplace=True)
        
    df_reg = df_reg.set_index('Cuenta')
    df_esf = df_esf.set_index('Cuenta')
    df_nat = df_nat.set_index('Cuenta')
    df_real = df_real.set_index('Cuenta')
    
    cuentas_comunes = list(set(df_reg.index) & set(df_esf.index) & set(df_nat.index) & set(df_real.index))
    
    # Convertir a numérico
    for df in [df_reg, df_esf, df_nat, df_real]:
        valid_cols = [c for c in sem_cols_20 if c in df.columns]
        df[valid_cols] = df[valid_cols].apply(pd.to_numeric, errors="coerce")

    # Arrays de resultados
    cuentas_validas = []
    reg_act, esf_act, nat_act, real_act = [], [], [], []
    semestres_termino_activos = []
    
    for cuenta in cuentas_comunes:
        esf_vals = df_esf.loc[cuenta, valid_cols].values.astype(float)
        
        # Ignorar trailing NaNs para no penalizar a los que aún no llegan al s20 o abandonaron definitivamente
        valid_idx = np.where(~np.isnan(esf_vals))[0]
        if len(valid_idx) == 0:
            continue
            
        last_valid = valid_idx[-1]
        esf_real = esf_vals[:last_valid+1]
        
        # Consideramos inactivo a los huecos (NaN) dentro del rango real
        num_inactivos = np.isnan(esf_real).sum()
        
        # Criterio: Rechazar si tiene > 4 semestres inactivos intermedios
        if num_inactivos > 4:
            continue
            
        cuentas_validas.append(cuenta)
        
        # Compactar cada dimensión (usando mask_active sobre los 20 valores para hacer el shift correctamente)
        mask_active = ~np.isnan(esf_vals)
        esf_shifted = esf_vals[mask_active]
        reg_shifted = df_reg.loc[cuenta, valid_cols].values.astype(float)[mask_active]
        nat_shifted = df_nat.loc[cuenta, valid_cols].values.astype(float)[mask_active]
        real_shifted = df_real.loc[cuenta, valid_cols].values.astype(float)[mask_active]
        
        # Padding hasta 20
        def pad(arr):
            return np.pad(arr, (0, max(0, 20 - len(arr))), constant_values=np.nan)[:20]
            
        reg_pad, esf_pad, nat_pad, real_pad = pad(reg_shifted), pad(esf_shifted), pad(nat_shifted), pad(real_shifted)
        
        reg_act.append(reg_pad)
        esf_act.append(esf_pad)
        nat_act.append(nat_pad)
        real_act.append(real_pad)
        
        # Calcular semestre_termino_activo
        # Buscar el primer 1 a partir de s8_activo (índice 7)
        s_term = np.nan
        for i in range(7, len(reg_pad)):
            if reg_pad[i] >= 1.0:
                s_term = i + 1 # 1-indexed
                break
        semestres_termino_activos.append(s_term)

    # Reconstruir DataFrames solo con s1 a s8 activos
    col_nombres = [f"s{i}" for i in range(1, 9)]
    
    def build_df(arr, suffix):
        df_new = pd.DataFrame(np.array(arr)[:, :8], columns=col_nombres, index=cuentas_validas)
        df_new = df_new.reset_index().rename(columns={'index': 'Cuenta'})
        return renombrar_semestres(df_new, suffix, 8)

    df_reg_final = build_df(reg_act, "reg")
    df_reg_final["semestre_termino"] = semestres_termino_activos
    
    df_esf_final = build_df(esf_act, "esf")
    df_nat_final = build_df(nat_act, "nat")
    df_real_final = build_df(real_act, "real")

    # Merge final
    df_final = (
        df_reg_final
        .merge(df_esf_final, on="Cuenta", how="inner")
        .merge(df_nat_final, on="Cuenta", how="inner")
        .merge(df_real_final, on="Cuenta", how="inner")
    )
    
    # Traer metadata (Generación) al df_final
    gen_col = encontrar_col_generacion(df_reg)
    if gen_col:
        df_meta = df_reg.loc[cuentas_validas, [gen_col]].reset_index()
        df_final = pd.merge(df_meta, df_final, on='Cuenta', how='right')
    
    # Limpiar columnas no deseadas que pudieran haber entrado por merge
    cols_a_borrar = [c for c in df_final.columns if c.endswith("_x") or c.endswith("_y")]
    df_final = df_final.drop(columns=cols_a_borrar, errors="ignore")

    nombre_salida = "Matemáticas" if carrera == "Matematicas" else "Física"
    df_final.to_excel(DATOS_LIMPIOS / f"{nombre_salida}.xlsx", index=False)
    print(f"  -> {nombre_salida} activos guardado con {len(df_final)} alumnos.")

# Test opcional para verificar funcionamiento
print("Ejecución finalizada.")
