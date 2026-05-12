"""
34_analisis_conglomerados.py
===========================
Ejecuta análisis de conglomerados (K-Means y Jerárquico) sobre las trayectorias
estudiantiles y valida si estos grupos naturales coinciden con los 4 cuadrantes
teóricos definidos por Regularidad y Promedio Natural (Élite, Pragmático, Perfeccionista, Rezago).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import scipy.cluster.hierarchy as sch

# ── Rutas y Configuración ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATOS_DIR = BASE_DIR / "Datos" / "Datos_limpios_activos"
IMG_DIR = BASE_DIR / "imagenes"
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", context="paper", font_scale=1.15)

# ── 1. CARGA DE DATOS Y DEFINICIÓN DE PERFILES ───────────────────────────────────
def load_data_and_profiles():
    dfs = []
    for carrera in ["Matemáticas", "Física"]:
        file_path = DATOS_DIR / f"{carrera}.xlsx"
        if file_path.exists():
            df = pd.read_excel(file_path)
            df["Carrera"] = carrera
            dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Usamos la trayectoria hasta S4 para definir el perfil base
    cols_analisis = [f"s{i}_{s}" for i in range(1, 5) for s in ["reg", "nat", "esf", "real"]]
    
    # Limpieza de NaNs en las columnas necesarias para el perfil
    df_clean = df_all.dropna(subset=["s4_reg", "s4_nat"] + cols_analisis).copy()
    
    # Definición de medianas poblacionales
    med_reg = df_clean["s4_reg"].median()
    med_nat = df_clean["s4_nat"].median()
    
    print(f"📊 Medianas al S4 -> Regularidad: {med_reg:.2f}, Promedio Natural: {med_nat:.2f}")
    
    # Función de perfil
    def get_profile(row):
        r = row["s4_reg"]
        n = row["s4_nat"]
        if r >= med_reg and n >= med_nat: return "Sin atraso"
        if r >= med_reg and n <  med_nat: return "Pragmático"
        if r <  med_reg and n >= med_nat: return "Perfeccionista"
        return "Rezago"
    
    df_clean["Perfil_Teorico"] = df_clean.apply(get_profile, axis=1)
    df_clean = df_clean.reset_index(drop=True) # Garantizar alineación con numpy arrays
    return df_clean, cols_analisis

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    print("🔬 Iniciando Validación de Grupos mediante Análisis de Conglomerados...")
    
    df, feat_cols = load_data_and_profiles()
    print(f"   Muestra limpia procesada: {len(df)} estudiantes.")
    print("   Distribución de Perfiles Teóricos:")
    print(df["Perfil_Teorico"].value_counts())

    # Estandarización
    X = df[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    
    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"✅ PCA calculado (Varianza explicada por PC1+PC2: {var_exp:.1f}%)")
    
    # ── SOLUCIÓN ESTADÍSTICA: Whitening (Estandarizar las PCs para igualar pesos) ──
    # Al forzar varianza unitaria en ambas, PC2 (Promedio) pesa igual que PC1 (Regularidad) en la distancia euclidiana
    scaler_pca = StandardScaler()
    X_for_cluster = scaler_pca.fit_transform(X_pca)
    print("⚖️ Componentes Principales Blanqueadas (Igualdad de pesos en distancias ejecutada).")

    # ── 2. MODELADO: CLUSTERING ─────────────────────────────────────────────────
    print("⚙️ Corriendo K-Means (K=4) sobre plano balanceado...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster_KMeans"] = kmeans.fit_predict(X_for_cluster)
    
    print("⚙️ Corriendo Jerárquico de Ward (4 ramas) sobre plano balanceado...")
    hclust = AgglomerativeClustering(n_clusters=4, linkage="ward")
    df["Cluster_Hierarchical"] = hclust.fit_predict(X_for_cluster)
    
    # Calcular Métricas de Acuerdo (Adjusted Rand Index)
    ari_km = adjusted_rand_score(df["Perfil_Teorico"], df["Cluster_KMeans"])
    ari_hi = adjusted_rand_score(df["Perfil_Teorico"], df["Cluster_Hierarchical"])
    
    print(f"\n📈 Índices Rand Ajustados (Grado de Coincidencia):")
    print(f"   K-Means vs Teórico: {ari_km:.4f}")
    print(f"   Jerárquico vs Teórico: {ari_hi:.4f}")

    # Paletas de color globales (usadas en dendrograma y scatter)
    pal_teorica = {
        "Sin atraso": "#2ecc71",     # Verde
        "Pragmático": "#3498db",     # Azul
        "Perfeccionista": "#f1c40f", # Amarillo
        "Rezago": "#e74c3c"          # Rojo
    }

    # ── NUEVO: Generación del Dendrograma (Árbol de Cercanías) ───────────────────
    print("🌳 Generando Dendrograma con Puntos de Grupo Real...")
    plt.figure(figsize=(14, 8))
    
    # Muestreo reproducible para consistencia visual
    sample_size = 150
    df_sample = df.sample(n=sample_size, random_state=42)
    X_sample = X_for_cluster[df_sample.index.values]
    
    linkage_matrix = sch.linkage(X_sample, method='ward')
    
    # Capturar diccionario del dendrograma para mapear las hojas
    ax_dendro = plt.gca()
    ddata = sch.dendrogram(linkage_matrix, truncate_mode=None, no_labels=True, color_threshold=6.0, ax=ax_dendro)
    
    # Dibujar línea de corte
    plt.axhline(y=6.0, color='black', linestyle='--', alpha=0.5, label='Corte K=4')
    
    # ── Mapeo Visual de Hojas a Grupos Teóricos ──
    # ddata['leaves'] nos da el orden de izquierda a derecha de los datos en la gráfica
    x_ticks = ax_dendro.get_xticks()
    ordered_leaves = ddata['leaves']
    
    # Para cada hoja graficada, pintamos un punto con su color TEÓRICO debajo de la base
    for x_coord, leaf_idx in zip(x_ticks, ordered_leaves):
        group_name = df_sample.iloc[leaf_idx]["Perfil_Teorico"]
        leaf_color = pal_teorica[group_name]
        plt.scatter(x_coord, -0.2, color=leaf_color, s=40, zorder=10) # Puntos bajo el eje Y=0

    plt.title("Árbol de Cercanías con Etiquetas de Grupo Teóricas (Base de Hojas)\n"
              "Observa cómo las ramas 'atrapan' visualmente el mismo color de base", fontsize=14)
    plt.ylabel("Distancia Euclidiana de Ward")
    plt.xlabel("Individuos (Puntos de color = Grupo Teórico Real)")
    
    # Leyenda de los puntos
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=10, label=k) 
                    for k, v in pal_teorica.items()]
    plt.legend(handles=custom_lines, title="Color Real del Estudiante", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dendrograma_agrupamiento.png", dpi=300)
    plt.close()
    print("✅ Dendrograma con codificación de color base guardado.")

    # ── 3. VISUALIZACIÓN COMPARATIVA ──────────────────────────────────────────────
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)
    
    # Gráfica 1: Perfiles Teóricos
    sns.scatterplot(
        data=df, x="PC1", y="PC2", hue="Perfil_Teorico", 
        palette=pal_teorica, alpha=0.6, s=25, ax=axes[0], edgecolor=None
    )
    axes[0].set_title("A) Perfiles Teóricos (Cuadrantes al S4)\n(Reg. x Promedio)", fontsize=14, fontweight='bold')
    axes[0].legend(title="Grupo", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    # Gráfica 2: K-Means
    sns.scatterplot(
        data=df, x="PC1", y="PC2", hue="Cluster_KMeans", 
        palette="Set2", alpha=0.6, s=25, ax=axes[1], edgecolor=None
    )
    axes[1].set_title(f"B) Grupos Naturales - K-Means (k=4)\n(ARI = {ari_km:.3f})", fontsize=14, fontweight='bold')
    axes[1].legend(title="Cluster ID", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)

    # Gráfica 3: Jerárquico
    sns.scatterplot(
        data=df, x="PC1", y="PC2", hue="Cluster_Hierarchical", 
        palette="Set1", alpha=0.6, s=25, ax=axes[2], edgecolor=None
    )
    axes[2].set_title(f"C) Grupos Naturales - Jerárquico Ward\n(ARI = {ari_hi:.3f})", fontsize=14, fontweight='bold')
    axes[2].legend(title="Rama ID", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)

    plt.suptitle("Validación de Grupos: Estructura Teórica vs. Conglomerados Naturales en Plano PCA", 
                 fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plot_name = "comparativa_clustering_vs_teorico.png"
    plt.savefig(IMG_DIR / plot_name, dpi=300, bbox_inches="tight")
    print(f"\n✅ Gráfica guardada en: {IMG_DIR / plot_name}")
    
    # ── 4. MATRICES DE CORRESPONDENCIA (VALIDACIÓN CUANTITATIVA) ──────────────────
    print("\n📋 Generando Tablas de Correspondencia Visuales...")
    
    ct_km = pd.crosstab(df["Perfil_Teorico"], df["Cluster_KMeans"], normalize='index') * 100
    ct_hi = pd.crosstab(df["Perfil_Teorico"], df["Cluster_Hierarchical"], normalize='index') * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(ct_km, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1)
    ax1.set_title("A) Correspondencia K-Means (% por Perfil Teórico)")
    ax1.set_ylabel("Perfil Teórico Original")
    ax1.set_xlabel("ID de Conglomerado Natural")
    
    sns.heatmap(ct_hi, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2)
    ax2.set_title("B) Correspondencia Árbol Jerárquico (% por Perfil Teórico)")
    ax2.set_ylabel("")
    ax2.set_xlabel("ID de Rama del Dendrograma")
    
    plt.suptitle("Matrices de Validación: Congruencia entre Teoría y Empiria Algorítmica", 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(IMG_DIR / "tabla_contingencia_clustering.png", dpi=300, bbox_inches="tight")
    print(f"✅ Nueva Gráfica Dual guardada en: {IMG_DIR / 'tabla_contingencia_clustering.png'}")
    
    print("\n🎉 Proceso finalizado exitosamente.")

if __name__ == "__main__":
    main()
