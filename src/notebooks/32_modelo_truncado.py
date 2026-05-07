"""
32_modelo_truncado.py
=====================
Enfoque Truncado: Regresión regularizada (Lasso / Ridge) para predecir
el semestre exacto de término en la submuestra sin censura (Generaciones ≤ 2016).

Al eliminar estudiantes sin semestre_termino (NaN), eliminamos el sesgo de censura
y podemos tratar la variable objetivo como continua.

Protocolo estadístico:
  - Normalización Z-score de predictores (necesaria para Lasso/Ridge).
  - Selección de alpha óptimo mediante 5-fold CV (RidgeCV / LassoCV).
  - IC al 95 % de RMSE, MAE y R² por bootstrapping (B = 2000).
  - Prueba F global del modelo vs. modelo nulo (H0: β=0 para todos los coefs).
  - Prueba t individual sobre coeficientes (Lasso: solo no-nulos).
  - Comparación Lasso vs. Ridge mediante prueba de hipótesis de Diebold-Mariano.
  - Exporta métricas a JSON y gráficas a imagenes/.

Uso:
  cd src/notebooks
  python3 32_modelo_truncado.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
DATOS_DIR  = BASE_DIR / "Datos" / "Datos_limpios_truncados"
IMG_DIR    = BASE_DIR / "imagenes"
OUTPUT_DIR = Path(__file__).resolve().parent

IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    dfs = []
    for nombre, carrera in [("Matemáticas.xlsx", "Matemáticas"), ("Física.xlsx", "Física")]:
        df = pd.read_excel(DATOS_DIR / nombre)
        df["Carrera"] = carrera
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["semestre_termino"])
    return df


def get_features(df: pd.DataFrame) -> list[str]:
    base = []
    for i in range(1, 9):
        for suf in ["reg", "esf", "nat", "real"]:
            col = f"s{i}_{suf}"
            if col in df.columns:
                base.append(col)
    return base


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, 9):
        if f"s{i}_esf" in df.columns and f"s{i}_real" in df.columns:
            df[f"s{i}_esf_x_real"] = df[f"s{i}_esf"] * df[f"s{i}_real"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP — IC para métricas de regresión
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmses, maes, r2s = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        maes.append(mean_absolute_error(yt, yp))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "rmse_mean":  float(np.mean(rmses)),
        "rmse_ci_lo": float(np.quantile(rmses, lo)),
        "rmse_ci_hi": float(np.quantile(rmses, hi)),
        "mae_mean":   float(np.mean(maes)),
        "mae_ci_lo":  float(np.quantile(maes, lo)),
        "mae_ci_hi":  float(np.quantile(maes, hi)),
        "r2_mean":    float(np.mean(r2s)),
        "r2_ci_lo":   float(np.quantile(r2s, lo)),
        "r2_ci_hi":   float(np.quantile(r2s, hi)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRUEBA F GLOBAL DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════
def f_test_global(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """
    H0: todos los coeficientes son cero (modelo nulo).
    F = [(SS_res_nulo - SS_res_mod) / p] / [SS_res_mod / (n - p - 1)]
    """
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    ss_reg = ss_tot - ss_res

    if (n - p - 1) <= 0 or ss_res == 0:
        return {"F": np.nan, "p_value": np.nan, "df1": p, "df2": n - p - 1}

    F = (ss_reg / p) / (ss_res / (n - p - 1))
    p_value = 1 - stats.f.cdf(F, dfn=p, dfd=n - p - 1)
    return {
        "F": float(F),
        "p_value": float(p_value),
        "df1": int(p),
        "df2": int(n - p - 1),
        "decision": "Rechazar H0 (p < 0.05): El modelo explica varianza significativa."
                    if p_value < 0.05 else
                    "No rechazar H0 (p >= 0.05): El modelo no mejora sobre el nulo.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRUEBA t SOBRE COEFICIENTES (Ridge / Lasso no-nulos)
# ═══════════════════════════════════════════════════════════════════════════════
def t_test_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    coef: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Aproximación de la varianza de los coeficientes usando la matriz de diseño
    y el MSE residual (válida asintóticamente para modelos regularizados).
    """
    n, p = X.shape
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / max(n - p - 1, 1)

    # Varianza-covarianza de OLS como aproximación
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        var_coef = mse * np.diag(XtX_inv)
        se = np.sqrt(np.abs(var_coef))
    except Exception:
        se = np.ones(len(coef)) * np.nan

    t_stats = coef / (se + 1e-12)
    df = max(n - p - 1, 1)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

    rows = []
    for name, c, s, t, pv in zip(feature_names, coef, se, t_stats, p_values):
        if c != 0.0:   # Solo mostrar coeficientes activos (relevante para Lasso)
            rows.append({
                "Variable": name,
                "Coef": round(float(c), 4),
                "SE": round(float(s), 4),
                "t": round(float(t), 3),
                "p_value": round(float(pv), 4),
                "Significativo": "✓" if pv < 0.05 else "✗",
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRUEBA DE DIEBOLD-MARIANO (Lasso vs. Ridge)
# ═══════════════════════════════════════════════════════════════════════════════
def diebold_mariano_test(
    y_true: np.ndarray,
    pred_lasso: np.ndarray,
    pred_ridge: np.ndarray,
) -> dict:
    """
    Compara los errores cuadráticos de dos modelos de regresión.
    H0: Los modelos tienen el mismo error de predicción esperado.
    """
    e_lasso = (y_true - pred_lasso) ** 2
    e_ridge = (y_true - pred_ridge) ** 2
    d = e_lasso - e_ridge          # positivo si Lasso es peor

    n = len(d)
    d_bar = np.mean(d)
    # Varianza de largo plazo de d (Newey-West con h=0 para simplicidad)
    var_d = np.var(d, ddof=1) / n
    dm_stat = d_bar / np.sqrt(var_d + 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_statistic": float(dm_stat),
        "p_value": float(p_value),
        "d_bar": float(d_bar),
        "decision": (
            "Rechazar H0 (p < 0.05): Los modelos difieren significativamente. "
            + ("Ridge es mejor." if d_bar > 0 else "Lasso es mejor.")
        ) if p_value < 0.05 else
        "No rechazar H0 (p >= 0.05): No hay diferencia significativa entre Lasso y Ridge.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(
    df: pd.DataFrame,
    features: list,
    model_name: str,
    B: int = 2000,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    df_m = df.dropna(subset=features + ["semestre_termino"]).copy()
    X_raw = df_m[features].values
    y     = df_m["semestre_termino"].values
    n, p  = X_raw.shape

    print(f"  N = {n} | p = {p} predictores")
    print(f"  y — mean: {y.mean():.2f}  std: {y.std():.2f}  "
          f"min: {y.min():.0f}  max: {y.max():.0f}")

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Ridge ──────────────────────────────────────────────────────────────────
    alphas_ridge = np.logspace(-3, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5, scoring="neg_mean_squared_error")
    ridge_cv.fit(X, y)
    alpha_ridge = ridge_cv.alpha_
    print(f"  Ridge — alpha óptimo (5-CV): {alpha_ridge:.4f}")

    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_ridge = cross_val_predict(ridge_cv, X, y, cv=cv5)
    bs_ridge = bootstrap_regression_metrics(y, y_pred_ridge, B=B)
    ftest_ridge = f_test_global(y, y_pred_ridge, p)
    print(f"  Ridge RMSE: {bs_ridge['rmse_mean']:.4f} [{bs_ridge['rmse_ci_lo']:.4f}, {bs_ridge['rmse_ci_hi']:.4f}]")
    print(f"  Ridge R²:   {bs_ridge['r2_mean']:.4f}  [{bs_ridge['r2_ci_lo']:.4f}, {bs_ridge['r2_ci_hi']:.4f}]")
    print(f"  F-test:     F={ftest_ridge['F']:.2f}, p={ftest_ridge['p_value']:.4e}")

    # Ajuste final para coeficientes
    ridge_cv.fit(X, y)
    coef_ridge = ridge_cv.coef_
    t_ridge = t_test_coefficients(X, y, ridge_cv.predict(X), coef_ridge, features)

    # ── Lasso ──────────────────────────────────────────────────────────────────
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=100)
    lasso_cv.fit(X, y)
    alpha_lasso = lasso_cv.alpha_
    print(f"  Lasso — alpha óptimo (5-CV): {alpha_lasso:.4f}")

    y_pred_lasso = cross_val_predict(lasso_cv, X, y, cv=cv5)
    bs_lasso = bootstrap_regression_metrics(y, y_pred_lasso, B=B)
    ftest_lasso = f_test_global(y, y_pred_lasso, p)
    n_active = np.sum(lasso_cv.coef_ != 0)
    print(f"  Lasso RMSE: {bs_lasso['rmse_mean']:.4f} [{bs_lasso['rmse_ci_lo']:.4f}, {bs_lasso['rmse_ci_hi']:.4f}]")
    print(f"  Lasso R²:   {bs_lasso['r2_mean']:.4f}  [{bs_lasso['r2_ci_lo']:.4f}, {bs_lasso['r2_ci_hi']:.4f}]")
    print(f"  Lasso coefs activos: {n_active}/{p}")

    lasso_cv.fit(X, y)
    coef_lasso = lasso_cv.coef_
    t_lasso = t_test_coefficients(X, y, lasso_cv.predict(X), coef_lasso, features)

    # ── Diebold-Mariano ────────────────────────────────────────────────────────
    dm = diebold_mariano_test(y, y_pred_lasso, y_pred_ridge)
    print(f"\n  Diebold-Mariano: DM={dm['DM_statistic']:.3f}, p={dm['p_value']:.4e}")
    print(f"  → {dm['decision']}")

    # Importancias (|coef| normalizado)
    imp_ridge = dict(zip(features, np.abs(coef_ridge).tolist()))
    imp_lasso = dict(zip(features, np.abs(coef_lasso).tolist()))

    return {
        "n_obs": int(n),
        "n_features": int(p),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "ridge": {
            "alpha": float(alpha_ridge),
            "bootstrap": bs_ridge,
            "f_test": ftest_ridge,
            "importances": imp_ridge,
            "n_active": int(p),
            "coef_table": t_ridge.to_dict(orient="records"),
        },
        "lasso": {
            "alpha": float(alpha_lasso),
            "bootstrap": bs_lasso,
            "f_test": ftest_lasso,
            "importances": imp_lasso,
            "n_active": int(n_active),
            "coef_table": t_lasso.to_dict(orient="records"),
        },
        "diebold_mariano": dm,
        "y_pred_ridge": y_pred_ridge.tolist(),
        "y_pred_lasso": y_pred_lasso.tolist(),
        "y_true": y.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pred_vs_real(result: dict, filename: str) -> None:
    y = np.array(result["y_true"])
    y_r = np.array(result["y_pred_ridge"])
    y_l = np.array(result["y_pred_lasso"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, yp, tag, color, bs in [
        (axes[0], y_r, "Ridge", "#3498db", result["ridge"]["bootstrap"]),
        (axes[1], y_l, "Lasso", "#e74c3c", result["lasso"]["bootstrap"]),
    ]:
        ax.scatter(y, yp, alpha=0.3, s=12, color=color, edgecolors="none")
        lims = [min(y.min(), yp.min()) - 0.5, max(y.max(), yp.max()) + 0.5]
        ax.plot(lims, lims, "k--", linewidth=1, label="Predicción perfecta")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Semestre Real", fontsize=11)
        ax.set_ylabel("Semestre Predicho", fontsize=11)
        ax.set_title(
            f"{tag} — Predicción vs. Real\n"
            f"RMSE: {bs['rmse_mean']:.3f} [IC: {bs['rmse_ci_lo']:.3f}–{bs['rmse_ci_hi']:.3f}]  "
            f"R²: {bs['r2_mean']:.3f}",
            fontsize=11,
        )
        ax.legend(fontsize=9)

    plt.suptitle("Enfoque Truncado: Predicción del Semestre de Término (Generaciones ≤ 2016)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


def plot_coef_importance(result: dict, model: str, top_n: int, filename: str) -> None:
    imp = pd.Series(result[model]["importances"]).sort_values(ascending=False).head(top_n)
    colors = [
        "#e74c3c" if "_esf_x_real" in f else
        "#3498db" if "_esf" in f else
        "#2ecc71" if "_reg" in f else
        "#f39c12"
        for f in imp.index
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.invert_yaxis()
    ax.set_title(f"Importancia de Variables ({model.capitalize()}) — Enfoque Truncado",
                 fontsize=12, pad=12)
    ax.set_xlabel("|Coeficiente| estandarizado", fontsize=10)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Interacción esf×real"),
        Patch(facecolor="#3498db", label="Índice de esfuerzo"),
        Patch(facecolor="#2ecc71", label="Regularidad"),
        Patch(facecolor="#f39c12", label="Promedio"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


def plot_metrics_comparison(result: dict, filename: str) -> None:
    metrics = [("rmse", "RMSE (↓ mejor)"), ("mae", "MAE (↓ mejor)"), ("r2", "R² (↑ mejor)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (m, lab) in zip(axes, metrics):
        for i, (mod, color) in enumerate([("ridge", "#3498db"), ("lasso", "#e74c3c")]):
            bs = result[mod]["bootstrap"]
            mean = bs[f"{m}_mean"]
            lo   = bs[f"{m}_ci_lo"]
            hi   = bs[f"{m}_ci_hi"]
            ax.bar(i, mean, color=color, alpha=0.85, width=0.5,
                   label=f"{mod.capitalize()} (α={result[mod]['alpha']:.4f})")
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="none", color="black", capsize=8, linewidth=2)
            ax.text(i, hi + 0.01 * abs(hi), f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(lab, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ridge", "Lasso"], fontsize=10)
        ax.set_ylabel("Valor (± IC 95%)", fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle("Comparativa Ridge vs. Lasso — IC al 95 % (Bootstrap, B=2000)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


def plot_residuals(result: dict, filename: str) -> None:
    y = np.array(result["y_true"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, color, label in [
        (axes[0], "y_pred_ridge", "#3498db", "Ridge"),
        (axes[1], "y_pred_lasso", "#e74c3c", "Lasso"),
    ]:
        yp = np.array(result[key])
        res = y - yp
        ax.scatter(yp, res, alpha=0.3, s=12, color=color, edgecolors="none")
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Semestre Predicho", fontsize=11)
        ax.set_ylabel("Residuo (Real − Predicho)", fontsize=11)
        ax.set_title(f"{label} — Gráfica de Residuos", fontsize=11)
        # Normalidad de residuos
        stat, pv = stats.shapiro(res[:min(len(res), 5000)])
        ax.text(0.05, 0.95, f"Shapiro-Wilk: p = {pv:.4f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.suptitle("Gráficas de Residuos — Enfoque Truncado", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n🔬 Iniciando pipeline — Enfoque Truncado (Lasso/Ridge)...")
    df = load_data()
    df = add_interactions(df)
    features = get_features(df) + [f"s{i}_esf_x_real" for i in range(1, 9)
                                    if f"s{i}_esf_x_real" in df.columns]

    print(f"\n📊 Total registros (Gen ≤ 2016, con semestre_termino): {len(df)}")
    print(f"   Distribución por carrera:\n{df['Carrera'].value_counts().to_string()}")
    print(f"   semestre_termino — media: {df['semestre_termino'].mean():.2f} "
          f"std: {df['semestre_termino'].std():.2f}")

    B = 2000
    result = train_model(df, features, "LASSO / RIDGE — Regresión Truncada", B=B)

    # ── Exportar JSON ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXPORTANDO MÉTRICAS")
    print(f"{'='*60}")
    output = {k: v for k, v in result.items()
              if k not in ("y_true", "y_pred_ridge", "y_pred_lasso")}
    json_path = OUTPUT_DIR / "metricas_modelo_truncado.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  → Guardado: {json_path.name}")

    # ── Gráficas ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GENERANDO GRÁFICAS")
    print(f"{'='*60}")
    plot_pred_vs_real(result, "trunc_pred_vs_real.png")
    plot_coef_importance(result, "ridge", top_n=12, filename="trunc_ridge_importancia.png")
    plot_coef_importance(result, "lasso", top_n=12, filename="trunc_lasso_importancia.png")
    plot_metrics_comparison(result, "trunc_comparativa_ic.png")
    plot_residuals(result, "trunc_residuos.png")

    print("\n✅ Pipeline Truncado completado exitosamente.")
    print(f"   Imágenes en: {IMG_DIR}")
    print(f"   Métricas en: {json_path}")


if __name__ == "__main__":
    main()
