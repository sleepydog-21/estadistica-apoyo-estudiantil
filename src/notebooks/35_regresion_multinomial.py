"""
35_regresion_multinomial.py
===========================
Sustituye los modelos Random Forest por Regresión Logística Multinomial (GLM).
Realiza clasificación estadística multivariante con regularización L2 (Ridge)
para modelar el estado final de los estudiantes.

Genera:
- Métricas de desempeño (Accuracy, F1) con IC Bootstrap.
- Coeficientes estandarizados (log-odds) para interpretación estadística.
- Matrices de Confusión y Comparativa McNemar.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
DATOS_DIR  = BASE_DIR / "Datos" / "Datos_limpios_multiclase"
IMG_DIR    = BASE_DIR / "imagenes"
OUTPUT_DIR = BASE_DIR / "src" / "notebooks"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Estilo ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS Y PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    dfs = []
    for nombre, carrera in [("Matemáticas.xlsx", "Matemáticas"), ("Física.xlsx", "Física")]:
        file_p = DATOS_DIR / nombre
        if file_p.exists():
            df = pd.read_excel(file_p)
            df["Carrera"] = carrera
            dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["estado_final"])
    return df_all

def build_interaction_features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, n + 1):
        if f"s{i}_esf" in df.columns and f"s{i}_real" in df.columns:
            df[f"s{i}_esf_x_real"] = df[f"s{i}_esf"] * df[f"s{i}_real"]
    return df

def get_feature_cols(n: int, df: pd.DataFrame) -> list[str]:
    base = []
    for i in range(1, n + 1):
        for suf in ["reg", "esf", "nat", "real"]:
            col = f"s{i}_{suf}"
            if col in df.columns:
                base.append(col)
        inter = f"s{i}_esf_x_real"
        if inter in df.columns:
            base.append(inter)
    return base

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP & TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap_metrics(y_true, y_pred, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    n = len(y_true)
    y_t_arr, y_p_arr = np.array(y_true), np.array(y_pred)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        accs.append(accuracy_score(y_t_arr[idx], y_p_arr[idx]))
        f1s.append(f1_score(y_t_arr[idx], y_p_arr[idx], average="macro", zero_division=0))
    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "accuracy_mean":  float(np.mean(accs)),
        "accuracy_ci_lo": float(np.quantile(accs, lo)),
        "accuracy_ci_hi": float(np.quantile(accs, hi)),
        "f1_macro_mean":  float(np.mean(f1s)),
        "f1_macro_ci_lo": float(np.quantile(f1s, lo)),
        "f1_macro_ci_hi": float(np.quantile(f1s, hi)),
    }

def bootstrap_per_class(y_true, y_pred, classes, B=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_t_arr, y_p_arr = np.array(y_true), np.array(y_pred)
    precs = {c: [] for c in classes}
    recs  = {c: [] for c in classes}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_t_arr[idx], y_p_arr[idx]
        for c in classes:
            tp = np.sum((yt == c) & (yp == c))
            fp = np.sum((yt != c) & (yp == c))
            fn = np.sum((yt == c) & (yp != c))
            precs[c].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recs[c].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    lo, hi = alpha / 2, 1 - alpha / 2
    res = {}
    for c in classes:
        res[c] = {
            "precision_mean":  float(np.mean(precs[c])),
            "precision_ci_lo": float(np.quantile(precs[c], lo)),
            "precision_ci_hi": float(np.quantile(precs[c], hi)),
            "recall_mean":     float(np.mean(recs[c])),
            "recall_ci_lo":    float(np.quantile(recs[c], lo)),
            "recall_ci_hi":    float(np.quantile(recs[c], hi)),
        }
    return res

def run_mcnemar(y_true, pred_A, pred_B):
    yt = np.array(y_true)
    pA = np.array(pred_A)
    pB = np.array(pred_B)
    cA = (pA == yt)
    cB = (pB == yt)
    b = np.sum(cA & ~cB)
    c = np.sum(~cA & cB)
    table = np.array([[np.sum(cA & cB), b], [c, np.sum(~cA & ~cB)]])
    res = mcnemar(table, exact=False, correction=True)
    return {
        "b": int(b), "c": int(c), "statistic": float(res.statistic), "p_value": float(res.pvalue)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 3. AJUSTE E INFERENCIA (REGRESIÓN LOGÍSTICA)
# ═══════════════════════════════════════════════════════════════════════════════
def train_logistic(df, n_sem, label, B=2000):
    print(f"\n🎯 Modelando {label} via Regresión Logística Multinomial Penalizada...")
    df = build_interaction_features(df, n_sem)
    feats = get_feature_cols(n_sem, df)
    df_m = df.dropna(subset=feats + ["estado_final"]).copy()
    
    X_raw = df_m[feats].values
    y = df_m["estado_final"].values
    classes = sorted(df_m["estado_final"].unique().tolist())
    
    # Estandarización obligatoria para coeficientes comparables y convergencia del solver
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # LogisticRegression rápida con C=1.0 fijo y iteraciones altas para evitar warnings
    clf = LogisticRegression(
        multi_class='multinomial',
        penalty='l2',
        solver='saga',
        C=1.0,
        max_iter=10000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Predicciones Out-Of-Fold
    cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_oof = cross_val_predict(clf, X, y, cv=cv_oof, method="predict", n_jobs=-1)
    
    print(f"📊 Accuracy OOF: {accuracy_score(y, y_pred_oof):.3f}")
    
    # Bootstrap
    bs = bootstrap_metrics(y, y_pred_oof, B=B)
    bs_cls = bootstrap_per_class(y, y_pred_oof, classes, B=B)
    
    # Ajuste final para obtener los coeficientes definitivos
    clf.fit(X, y)
    
    # En multinomial, clf.coef_ tiene forma (n_clases, n_features)
    coef_dict = {}
    for i, cls_name in enumerate(clf.classes_):
        coef_dict[cls_name] = dict(zip(feats, clf.coef_[i].tolist()))
        
    return {
        "label": label, "n_sem": n_sem, "classes": classes, "features": feats,
        "coefs": coef_dict, "confusion_matrix": confusion_matrix(y, y_pred_oof, labels=classes).tolist(),
        "bootstrap": bs, "bootstrap_per_class": bs_cls, "y_true": y, "y_pred_oof": y_pred_oof, "c_opt": 1.0
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICAS ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cm(res, fname):
    cm = np.array(res["confusion_matrix"])
    cls = res["classes"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cls, yticklabels=cls, ax=ax)
    ax.set_title(f"Matriz de Confusión - Logística Multinomial ({res['label']})")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_DIR / fname, dpi=300)
    plt.close()

def plot_coefficients(res, top_k=10, fname="coefs.png"):
    """Grafica los coeficientes beta estandarizados para ver el efecto."""
    coefs = res["coefs"]
    classes = res["classes"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False)
    axes = axes.flatten()
    
    for idx, cname in enumerate(classes):
        if idx >= len(axes): break
        ax = axes[idx]
        # Obtener los K coeficientes con mayor magnitud absoluta
        ser = pd.Series(coefs[cname])
        ser_top = ser.reindex(ser.abs().sort_values(ascending=False).head(top_k).index)
        ser_top = ser_top.iloc[::-1] # invertir para graficar
        
        # Colores por signo
        colors = ["#2ecc71" if val > 0 else "#e74c3c" for val in ser_top.values]
        ser_top.plot(kind='barh', ax=ax, color=colors, edgecolor='white', alpha=0.8)
        
        ax.set_title(f"Efecto Log-Odds: {cname}", fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Beta Estandarizado")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
    plt.suptitle(f"Impacto de Covariables en Probabilidad Logit ({res['label']})", fontsize=16, y=0.99, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(IMG_DIR / fname, dpi=300)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("🔬 Iniciando estimación de Modelos Logísticos Multinomiales...")
    df_all = load_data()
    print(f"✅ Registros cargados: {len(df_all)}")
    
    B_REPS = 2000
    
    # --- Modelo A (S1-S4) ---
    res_A = train_logistic(df_all, 4, "Modelo A (S1-S4)", B=B_REPS)
    
    # --- Modelo B (S1-S8) ---
    res_B = train_logistic(df_all, 8, "Modelo B (S1-S8)", B=B_REPS)
    
    # --- McNemar ---
    print("\n📊 Realizando Test de McNemar...")
    # Extraemos el dataframe filtrado de nuevo para intersectar índices
    d_A = build_interaction_features(df_all, 4).dropna(subset=get_feature_cols(4, df_all)+["estado_final"]).reset_index(drop=True)
    d_B = build_interaction_features(df_all, 8).dropna(subset=get_feature_cols(8, df_all)+["estado_final"]).reset_index(drop=True)
    common = d_A.index.intersection(d_B.index)
    
    yt_c = d_A.loc[common, "estado_final"].values
    pA_c = np.array(res_A["y_pred_oof"])[common]
    pB_c = np.array(res_B["y_pred_oof"])[common]
    
    mcn = run_mcnemar(yt_c, pA_c, pB_c)
    print(f"📈 McNemar p-valor: {mcn['p_value']:.4e}")
    
    # --- Guardar Resultados ---
    print("\n💾 Guardando gráficas y métricas...")
    
    plot_cm(res_A, "logit_cm_s4.png")
    plot_cm(res_B, "logit_cm_s8.png")
    plot_coefficients(res_A, top_k=12, fname="logit_coefs_s4.png")
    plot_coefficients(res_B, top_k=12, fname="logit_coefs_s8.png")
    
    # Exportar resumen a JSON para LaTeX
    output = {
        "modelo_A": {
            "accuracy": res_A["bootstrap"]["accuracy_mean"],
            "f1": res_A["bootstrap"]["f1_macro_mean"],
            "c_optimo": float(res_A["c_opt"])
        },
        "modelo_B": {
            "accuracy": res_B["bootstrap"]["accuracy_mean"],
            "f1": res_B["bootstrap"]["f1_macro_mean"],
            "c_optimo": float(res_B["c_opt"])
        },
        "mcnemar": mcn
    }
    with open(OUTPUT_DIR / "metricas_logit_multinomial.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("\n🎉 Modelado Estadístico Logit Finalizado.")
    print(f"   Gráficas generadas en: {IMG_DIR}")
    print(f"   - logit_coefs_s4.png (Importancia de covariables)")
    print(f"   - logit_coefs_s8.png")

if __name__ == "__main__":
    main()
