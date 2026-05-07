"""
29_modelos_predictivos.py
=========================
Entrena dos modelos Random Forest para predecir el estado académico final:

  Modelo A  — Alerta Temprana  : predictores S1-S4  (sistema de alerta a mitad de carrera)
  Modelo B  — Trayectoria Full : predictores S1-S8  (toda la trayectoria disponible)

Para cada modelo calcula:
  - Accuracy e IC al 95 % por bootstrap (B=2000 reps)
  - F1-Score macro e IC al 95 % por bootstrap
  - Precisión y Recall por clase con IC
  - Prueba de hipótesis McNemar (Modelo A vs Modelo B)
  - Exporta métricas a JSON y gráficas a imagenes/

Uso:
  cd src/notebooks
  python 29_modelos_predictivos.py
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
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
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
OUTPUT_DIR = BASE_DIR / "src" / "notebooks"      # JSON de métricas aquí

IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta y estilo ────────────────────────────────────────────────────────────
PALETTE = {
    "Titulado a tiempo":    "#2ecc71",
    "Titulado con rezago":  "#f39c12",
    "Abandono silencioso":  "#e74c3c",
    "Activo / Aún cursando":"#3498db",
}
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
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["estado_final"])
    return df_all


def build_interaction_features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Agrega variables de interacción esf × real para S1..Sn."""
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
# 2. BOOTSTRAP — Intervalos de Confianza
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrapping sobre predicciones ya hechas (out-of-fold)."""
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "accuracy_mean":  float(np.mean(accs)),
        "accuracy_ci_lo": float(np.quantile(accs, lo)),
        "accuracy_ci_hi": float(np.quantile(accs, hi)),
        "f1_macro_mean":  float(np.mean(f1s)),
        "f1_macro_ci_lo": float(np.quantile(f1s, lo)),
        "f1_macro_ci_hi": float(np.quantile(f1s, hi)),
    }


def bootstrap_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    precs = {c: [] for c in classes}
    recs  = {c: [] for c in classes}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        for c in classes:
            tp = np.sum((yt == c) & (yp == c))
            fp = np.sum((yt != c) & (yp == c))
            fn = np.sum((yt == c) & (yp != c))
            precs[c].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recs[c].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    lo, hi = alpha / 2, 1 - alpha / 2
    result = {}
    for c in classes:
        result[c] = {
            "precision_mean":  float(np.mean(precs[c])),
            "precision_ci_lo": float(np.quantile(precs[c], lo)),
            "precision_ci_hi": float(np.quantile(precs[c], hi)),
            "recall_mean":     float(np.mean(recs[c])),
            "recall_ci_lo":    float(np.quantile(recs[c], lo)),
            "recall_ci_hi":    float(np.quantile(recs[c], hi)),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRUEBA DE HIPÓTESIS: McNEMAR
# ═══════════════════════════════════════════════════════════════════════════════
def mcnemar_test(y_true, pred_A, pred_B) -> dict:
    """
    H0: Modelo A y Modelo B tienen la misma tasa de error.
    Tabla de contingencia 2×2 sobre pares de predicciones.
    """
    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    # b: A bien, B mal  |  c: A mal, B bien
    b = np.sum(correct_A & ~correct_B)
    c = np.sum(~correct_A & correct_B)

    table = np.array([[np.sum(correct_A & correct_B), b],
                      [c, np.sum(~correct_A & ~correct_B)]])

    result = mcnemar(table, exact=False, correction=True)
    return {
        "b": int(b), "c": int(c),
        "statistic": float(result.statistic),
        "p_value":   float(result.pvalue),
        "decision":  "Rechazar H0 (p < 0.05): Los modelos difieren significativamente."
                     if result.pvalue < 0.05
                     else "No rechazar H0 (p >= 0.05): No hay diferencia significativa entre modelos.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO CON CROSS-VALIDATION (OUT-OF-FOLD)
# ═══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(
    df: pd.DataFrame,
    n_sem: int,
    label: str,
    B: int = 2000,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  MODELO {label} — S1 a S{n_sem}")
    print(f"{'='*60}")

    df = build_interaction_features(df, n_sem)
    features = get_feature_cols(n_sem, df)
    df_m = df.dropna(subset=features + ["estado_final"]).copy()

    X = df_m[features].values
    y = df_m["estado_final"].values
    classes = sorted(df_m["estado_final"].unique().tolist())

    print(f"  Observaciones: {len(df_m)} | Clases: {classes}")
    print(f"  Distribución:\n{pd.Series(y).value_counts().to_string()}\n")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # OOF via 5-fold estratificado
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_oof = cross_val_predict(rf, X, y, cv=cv, method="predict", n_jobs=-1)

    print("Classification Report (OOF):")
    print(classification_report(y, y_pred_oof, zero_division=0))

    # Bootstrap
    print(f"  Calculando IC al 95 % con B={B} repeticiones...")
    bs = bootstrap_metrics(y, y_pred_oof, B=B)
    bs_cls = bootstrap_per_class(y, y_pred_oof, classes, B=B)

    print(f"  Accuracy: {bs['accuracy_mean']:.4f}  [{bs['accuracy_ci_lo']:.4f}, {bs['accuracy_ci_hi']:.4f}]")
    print(f"  F1-macro: {bs['f1_macro_mean']:.4f}  [{bs['f1_macro_ci_lo']:.4f}, {bs['f1_macro_ci_hi']:.4f}]")

    # Ajuste final en todo el dataset para obtener importancias
    rf.fit(X, y)

    result = {
        "label":        label,
        "n_semestres":  n_sem,
        "n_obs":        len(df_m),
        "classes":      classes,
        "features":     features,
        "importances":  dict(zip(features, rf.feature_importances_.tolist())),
        "confusion_matrix": confusion_matrix(y, y_pred_oof, labels=classes).tolist(),
        "bootstrap":    bs,
        "bootstrap_per_class": bs_cls,
        "y_true":       y.tolist(),
        "y_pred_oof":   y_pred_oof.tolist(),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(result: dict, filename: str) -> None:
    cm = np.array(result["confusion_matrix"])
    classes = result["classes"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=0.5, linecolor="lightgray",
    )
    ax.set_title(
        f"Matriz de Confusión — {result['label']} (S1-S{result['n_semestres']})\n"
        f"Accuracy: {result['bootstrap']['accuracy_mean']:.3f} "
        f"[IC 95 %: {result['bootstrap']['accuracy_ci_lo']:.3f}–"
        f"{result['bootstrap']['accuracy_ci_hi']:.3f}]",
        fontsize=12, pad=14,
    )
    ax.set_ylabel("Etiqueta Verdadera", fontsize=11)
    ax.set_xlabel("Etiqueta Predicha", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


def plot_feature_importance(result: dict, top_n: int, filename: str) -> None:
    imp = pd.Series(result["importances"]).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#e74c3c" if "_esf_x_real" in f else
        "#3498db" if "_esf" in f else
        "#2ecc71" if "_reg" in f else
        "#f39c12"
        for f in imp.index
    ]
    imp.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.invert_yaxis()
    ax.set_title(
        f"Top {top_n} Predictores — {result['label']} (S1-S{result['n_semestres']})",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Importancia (Gini)", fontsize=11)
    # Leyenda de colores
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


def plot_ci_comparison(result_A: dict, result_B: dict, filename: str) -> None:
    """Gráfica de barras con IC del Accuracy y F1-macro para ambos modelos."""
    metrics = ["accuracy", "f1_macro"]
    labels  = ["Accuracy", "F1-Score Macro"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, m, lab in zip(axes, metrics, labels):
        for i, (res, color) in enumerate([(result_A, "#3498db"), (result_B, "#e74c3c")]):
            mean = res["bootstrap"][f"{m}_mean"]
            lo   = res["bootstrap"][f"{m}_ci_lo"]
            hi   = res["bootstrap"][f"{m}_ci_hi"]
            ax.bar(i, mean, color=color, alpha=0.8, width=0.5,
                   label=f"{res['label']} (S1-S{res['n_semestres']})")
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="none", color="black", capsize=8, linewidth=2)
            ax.text(i, hi + 0.005, f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(lab, fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Modelo A\nS1-S4", f"Modelo B\nS1-S8"], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Valor (± IC 95 %)", fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle("Comparativa de Modelos con Intervalos de Confianza al 95 % (Bootstrap)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n🔬 Iniciando pipeline de modelos predictivos...")
    df_all = load_data()
    print(f"\n📊 Total de registros cargados: {len(df_all)}")
    print(f"   Distribución de clases:\n{df_all['estado_final'].value_counts().to_string()}")

    BOOTSTRAP_REPS = 2000

    # ── Modelo A: Alerta Temprana (S1-S4) ─────────────────────────────────────
    result_A = train_and_evaluate(df_all, n_sem=4, label="A (Alerta Temprana)", B=BOOTSTRAP_REPS)

    # ── Modelo B: Trayectoria Completa (S1-S8) ────────────────────────────────
    result_B = train_and_evaluate(df_all, n_sem=8, label="B (Trayectoria Completa)", B=BOOTSTRAP_REPS)

    # ── Prueba de hipótesis McNemar ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PRUEBA DE HIPÓTESIS: McNEMAR (A vs B)")
    print(f"{'='*60}")

    # Alinear por índice: usar sólo registros presentes en AMBOS modelos
    df_A = df_all.copy()
    df_A = build_interaction_features(df_A, 4)
    df_A = df_A.dropna(subset=get_feature_cols(4, df_A) + ["estado_final"]).reset_index(drop=True)

    df_B = df_all.copy()
    df_B = build_interaction_features(df_B, 8)
    df_B = df_B.dropna(subset=get_feature_cols(8, df_B) + ["estado_final"]).reset_index(drop=True)

    common_idx = df_A.index.intersection(df_B.index)
    yt_common  = df_A.loc[common_idx, "estado_final"].values
    pA_common  = np.array(result_A["y_pred_oof"])[common_idx]
    pB_common  = np.array(result_B["y_pred_oof"])[common_idx]

    mc = mcnemar_test(yt_common, pA_common, pB_common)
    print(f"  b (A bien, B mal): {mc['b']}   c (A mal, B bien): {mc['c']}")
    print(f"  Estadístico χ²: {mc['statistic']:.4f}")
    print(f"  P-valor: {mc['p_value']:.4e}")
    print(f"  → {mc['decision']}")

    # ── Exportar métricas a JSON ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXPORTANDO MÉTRICAS A JSON")
    print(f"{'='*60}")

    output = {
        "modelo_A": {k: v for k, v in result_A.items() if k not in ("y_true", "y_pred_oof")},
        "modelo_B": {k: v for k, v in result_B.items() if k not in ("y_true", "y_pred_oof")},
        "mcnemar": mc,
    }
    json_path = OUTPUT_DIR / "metricas_modelos_predictivos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  → Métricas guardadas en: {json_path.name}")

    # ── Gráficas ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GENERANDO GRÁFICAS")
    print(f"{'='*60}")

    plot_confusion_matrix(result_A, "rf_confusion_matrix_s4.png")
    plot_confusion_matrix(result_B, "rf_confusion_matrix_s8.png")
    plot_feature_importance(result_A, top_n=12, filename="rf_feature_importance_s4.png")
    plot_feature_importance(result_B, top_n=12, filename="rf_feature_importance_s8.png")
    plot_ci_comparison(result_A, result_B, filename="rf_comparativa_ic.png")

    print("\n✅ Pipeline completado exitosamente.")
    print(f"   Imágenes en: {IMG_DIR}")
    print(f"   Métricas en: {json_path}")


if __name__ == "__main__":
    main()
