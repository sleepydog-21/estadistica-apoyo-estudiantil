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

    # ── NUEVO: Generación del Dendrograma Avanzado con Codificación Teórica ───────
    print("🌳 Generando Dendrograma Codificado por Grupo Teórico (Pintado Recursivo)...")
    from collections import Counter
    
    plt.figure(figsize=(14, 8))
    
    # Muestreo reproducible y alineado con la matriz PCA ya blanqueada
    sample_size = 180 # Aumentamos un poco el tamaño para que se aprecie bien la estructura
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    # Importante: Usamos los índices originales de df para extraer el PCA correspondiente,
    # pero luego reseteamos el index de df_sample para que el bucle coincida 1-a-1 con la matriz X_sample.
    numeric_indices = df_sample.index.values
    X_sample = X_for_cluster[numeric_indices]
    df_sample = df_sample.reset_index(drop=True)
    
    # 1. Calcular matriz de enlace
    linkage_matrix = sch.linkage(X_sample, method='ward')
    
    # 2. Preparar estructuras para el algoritmo de coloración por moda recursiva
    # Guardamos los perfiles individuales de cada hoja (nodo original)
    n_sample = len(df_sample)
    node_members = [[df_sample.iloc[i]["Perfil_Teorico"]] for i in range(n_sample)]
    
    # Inicializamos el diccionario de colores con los colores exactos de las hojas
    color_lookup = [pal_teorica[df_sample.iloc[i]["Perfil_Teorico"]] for i in range(n_sample)]
    
    # 3. Recorrer la matriz de enlace y determinar el color dominante de cada merge
    for i in range(len(linkage_matrix)):
        c1 = int(linkage_matrix[i, 0])
        c2 = int(linkage_matrix[i, 1])
        
        # El nuevo nodo hereda todos los miembros de sus hijos
        combined_members = node_members[c1] + node_members[c2]
        node_members.append(combined_members)
        
        # Calculamos la moda estadística (el grupo dominante debajo de este enlace)
        mode_group = Counter(combined_members).most_common(1)[0][0]
        color_lookup.append(pal_teorica[mode_group])
    
    # 4. Graficar el dendrograma usando la función de mapeo forzado
    # below_threshold_color y above_threshold_color no importan porque link_color_func sobreescribe todo
    ddata = sch.dendrogram(
        linkage_matrix, 
        no_labels=True, 
        link_color_func=lambda k: color_lookup[k]
    )
    
    plt.title("Mapa Estructural de Trayectorias Académicas\n"
              "(Ramas coloreadas según el Grupo Teórico Dominante en la fusión)", fontsize=14, pad=15)
    plt.ylabel("Distancia Euclidiana de Ward (PCA Blanqueado)")
    plt.xlabel("Individuos Agrupados Jerárquicamente")
    
    # Leyenda oficial
    from matplotlib.lines import Line2D
    custom_legend = [Line2D([0], [0], color=v, lw=3, label=k) 
                     for k, v in pal_teorica.items()]
    plt.legend(handles=custom_legend, title="Grupo Teórico Dominante", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dendrograma_agrupamiento.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Dendrograma codificado recursivamente guardado exitosamente.")

    # ── NUEVO: Dendrograma Separado por Carrera (FULL DATOS, TODA LA POBLACIÓN) ──
    print("🌳 Generando Dendrogramas de Población Completa por Carrera...")
    fig_carr, axes_carr = plt.subplots(1, 2, figsize=(26, 12))
    
    for ax_idx, carrera in enumerate(["Matemáticas", "Física"]):
        # Seleccionamos ABSOLUTAMENTE TODOS los registros de esa carrera
        df_sub = df[df["Carrera"] == carrera].copy()
        total_n = len(df_sub)
        print(f"   -> Procesando {total_n} estudiantes para {carrera}...")
        
        # Alinear matrices usando el dataset completo sin muestreo
        numeric_sub_indices = df_sub.index.values
        X_sub = X_for_cluster[numeric_sub_indices]
        df_sub = df_sub.reset_index(drop=True)
        
        # 1. Calcular matriz de enlace completa
        Z_sub = sch.linkage(X_sub, method='ward')
        
        # 2. ESTRUCTURA OPTIMIZADA DE ALTA ESCALA: Usamos Contadores (Frecuencias) en vez de Listas
        # Esto previene el consumo de memoria O(N^2) y permite procesar miles de registros al instante.
        sub_node_counters = [Counter([df_sub.iloc[i]["Perfil_Teorico"]]) for i in range(total_n)]
        sub_color_lookup = [pal_teorica[df_sub.iloc[i]["Perfil_Teorico"]] for i in range(total_n)]
        
        # 3. Agregación ultra-rápida de frecuencias dominantes
        for i in range(len(Z_sub)):
            c1 = int(Z_sub[i, 0])
            c2 = int(Z_sub[i, 1])
            
            # Sumar contadores es extremadamente eficiente y no consume memoria redundante
            combined_counter = sub_node_counters[c1] + sub_node_counters[c2]
            sub_node_counters.append(combined_counter)
            
            # Extraer la moda del contador consolidado
            mode_group = combined_counter.most_common(1)[0][0]
            sub_color_lookup.append(pal_teorica[mode_group])
            
        # 4. Graficar en el eje con líneas un poco más gruesas para visibilidad
        import matplotlib as mpl
        orig_lw = mpl.rcParams['lines.linewidth']
        mpl.rcParams['lines.linewidth'] = 0.8 # Aumentado de 0.4 a 0.8 por petición de visibilidad
        
        sch.dendrogram(
            Z_sub, 
            no_labels=True, 
            link_color_func=lambda k: sub_color_lookup[k],
            ax=axes_carr[ax_idx]
        )
        mpl.rcParams['lines.linewidth'] = orig_lw # Restaurar original
        axes_carr[ax_idx].set_title(f"{carrera}\n(Población Total: {total_n} registros)", fontsize=16, fontweight='bold')
        axes_carr[ax_idx].set_ylabel("Distancia Euclidiana (Ward)")
        
    # Título y leyendas globales
    fig_carr.suptitle("Topología Completa de Trayectorias Académicas por Licenciatura\n"
                      "Mapeo Exhaustivo de la Dominancia del Grupo Teórico (N Total)", fontsize=20, fontweight='bold', y=1.02)
    
    legend_lines = [Line2D([0], [0], color=v, lw=5, label=k) for k, v in pal_teorica.items()]
    fig_carr.legend(handles=legend_lines, title="Grupo Teórico Dominante", loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.94), fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(IMG_DIR / "dendrograma_por_carrera.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Dendrogramas Poblacionales Completos guardados exitosamente.")

    # ── NUEVO: Validación de Estabilidad (Muestra Balanceada Equitativa) ──────────
    print("\n⚖️ Iniciando Prueba de Estabilidad: Muestreo Balanceado Equitativo...")
    
    # 1. Determinar el tamaño máximo posible para muestras equilibradas
    conteo_grupos = df["Perfil_Teorico"].value_counts()
    min_n_per_group = conteo_grupos.min()
    print(f"   -> Tamaño de balanceo fijado en {min_n_per_group} registros por grupo.")
    
    # 2. Construir dataset balanceado (muestreo de igual tamaño para cada uno de los 4)
    dfs_balanceados = []
    for grupo in pal_teorica.keys():
        sub_df = df[df["Perfil_Teorico"] == grupo].sample(n=min_n_per_group, random_state=42)
        dfs_balanceados.append(sub_df)
        
    df_bal = pd.concat(dfs_balanceados).sample(frac=1, random_state=42).reset_index(drop=True)
    n_bal = len(df_bal)
    print(f"   -> Dataset balanceado construido con {n_bal} registros totales.")
    
    # 3. Extraer componentes blanqueadas
    indices_bal = df_bal.index.values # Al usar df_bal reconstruido, debemos tomar el subset original de nuevo?
    # ¡Corrección! Para alinear con X_for_cluster, necesitamos los índices originales que el dataframe traía.
    # Haremos el reset_index al final.
    df_bal_raw = pd.concat(dfs_balanceados).sample(frac=1, random_state=42)
    orig_bal_indices = df_bal_raw.index.values
    X_bal = X_for_cluster[orig_bal_indices]
    df_bal_final = df_bal_raw.reset_index(drop=True)
    
    # 4. Calcular Linkage y Coloración
    Z_bal = sch.linkage(X_bal, method='ward')
    bal_node_counters = [Counter([df_bal_final.iloc[i]["Perfil_Teorico"]]) for i in range(n_bal)]
    bal_color_lookup = [pal_teorica[df_bal_final.iloc[i]["Perfil_Teorico"]] for i in range(n_bal)]
    
    for i in range(len(Z_bal)):
        c1, c2 = int(Z_bal[i, 0]), int(Z_bal[i, 1])
        combined = bal_node_counters[c1] + bal_node_counters[c2]
        bal_node_counters.append(combined)
        bal_color_lookup.append(pal_teorica[combined.most_common(1)[0][0]])
        
    # 5. Graficar Dendrograma Balanceado
    plt.figure(figsize=(16, 9))
    import matplotlib as mpl
    orig_lw = mpl.rcParams['lines.linewidth']
    mpl.rcParams['lines.linewidth'] = 0.8 # Líneas visibles
    
    sch.dendrogram(
        Z_bal,
        no_labels=True,
        link_color_func=lambda k: bal_color_lookup[k]
    )
    mpl.rcParams['lines.linewidth'] = orig_lw
    
    plt.title("Prueba de Estabilidad Arquitectónica: Árbol Equilibrado (Muestra Balanceada)\n"
              f"Población homogeneizada a N={min_n_per_group} estudiantes por grupo teórico (Total N={n_bal})", 
              fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("Distancia Euclidiana (Ward)")
    plt.xlabel("Trayectorias Normalizadas por Volumen de Grupo")
    
    custom_leg = [Line2D([0], [0], color=v, lw=4, label=f"{k} (n={min_n_per_group})") for k, v in pal_teorica.items()]
    plt.legend(handles=custom_leg, title="Grupos Equilibrados", loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dendrograma_balanceado_total.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Dendrograma Balanceado de Validación guardado exitosamente.")

    # ── NUEVO: Validación de Estabilidad Balanceada POR CARRERA ──────────────────
    print("\n⚖️ Iniciando Prueba de Estabilidad Balanceada SEGMENTADA POR CARRERA...")
    fig_b, axes_b = plt.subplots(1, 2, figsize=(26, 12))
    
    for ax_idx, carrera in enumerate(["Matemáticas", "Física"]):
        df_c = df[df["Carrera"] == carrera].copy()
        
        # Obtener el límite de muestreo de ESTA carrera en particular
        local_counts = df_c["Perfil_Teorico"].value_counts()
        c_min = local_counts.min()
        print(f"   -> {carrera}: Muestreando balanceo a {c_min} estudiantes por grupo...")
        
        # Construir subset balanceado de la carrera
        dfs_c_bal = []
        for gp in pal_teorica.keys():
            subset_g = df_c[df_c["Perfil_Teorico"] == gp].sample(n=c_min, random_state=42)
            dfs_c_bal.append(subset_g)
        
        df_cb_raw = pd.concat(dfs_c_bal).sample(frac=1, random_state=42)
        cb_orig_indices = df_cb_raw.index.values
        X_cb = X_for_cluster[cb_orig_indices]
        df_cb_final = df_cb_raw.reset_index(drop=True)
        n_cb = len(df_cb_final)
        
        # Linkage y Coloración
        Z_cb = sch.linkage(X_cb, method='ward')
        cb_node_cnt = [Counter([df_cb_final.iloc[i]["Perfil_Teorico"]]) for i in range(n_cb)]
        cb_color_map = [pal_teorica[df_cb_final.iloc[i]["Perfil_Teorico"]] for i in range(n_cb)]
        
        for i in range(len(Z_cb)):
            c1, c2 = int(Z_cb[i, 0]), int(Z_cb[i, 1])
            comb = cb_node_cnt[c1] + cb_node_cnt[c2]
            cb_node_cnt.append(comb)
            cb_color_map.append(pal_teorica[comb.most_common(1)[0][0]])
            
        # Graficar
        orig_lw = mpl.rcParams['lines.linewidth']
        mpl.rcParams['lines.linewidth'] = 0.8 # Líneas bien visibles
        sch.dendrogram(Z_cb, no_labels=True, link_color_func=lambda k: cb_color_map[k], ax=axes_b[ax_idx])
        mpl.rcParams['lines.linewidth'] = orig_lw
        
        axes_b[ax_idx].set_title(f"{carrera}\n(Balanceado: {c_min} por grupo, N={n_cb})", fontsize=18, fontweight='bold')
        axes_b[ax_idx].set_ylabel("Distancia Euclidiana (Ward)")
        
    fig_b.suptitle("Comparativa de Estabilidad Topológica Balanceada por Carrera\n"
                   "Evaluación de Autonomía Estructural Eliminando el Sesgo de Población", 
                   fontsize=22, fontweight='bold', y=1.03)
    
    final_legend = [Line2D([0], [0], color=v, lw=6, label=k) for k, v in pal_teorica.items()]
    fig_b.legend(handles=final_legend, title="Perfiles Equitativos (25% cada uno)", 
                 loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(IMG_DIR / "dendrograma_balanceado_por_carrera.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Comparativa Balanceada por Carrera guardada exitosamente.")

    # ── NUEVO: Visualización de Plano PCA Balanceado por Carrera ──────────────────
    print("\n🗺️ Generando Mapa PCA de Densidad Balanceada por Carrera...")
    fig_pca_bal, axes_pca = plt.subplots(1, 2, figsize=(20, 8), sharey=True, sharex=True)
    
    for ax_idx, carrera in enumerate(["Matemáticas", "Física"]):
        df_c = df[df["Carrera"] == carrera].copy()
        local_counts = df_c["Perfil_Teorico"].value_counts()
        c_min = local_counts.min()
        
        # Samplear balanceado para la gráfica
        dfs_pca_bal = []
        for gp in pal_teorica.keys():
            subset_g = df_c[df_c["Perfil_Teorico"] == gp].sample(n=c_min, random_state=42)
            dfs_pca_bal.append(subset_g)
        
        # Mezclar para que los colores en la gráfica se interpolen bien visualmente
        df_c_balanced = pd.concat(dfs_pca_bal).sample(frac=1, random_state=42)
        n_total_plot = len(df_c_balanced)
        
        # Dibujar los puntos con transparencias
        sns.scatterplot(
            data=df_c_balanced, 
            x="PC1", y="PC2", 
            hue="Perfil_Teorico", 
            palette=pal_teorica, 
            alpha=0.6, s=35, 
            ax=axes_pca[ax_idx], 
            edgecolor=None
        )
        
        axes_pca[ax_idx].set_title(f"{carrera}\n(N={c_min} puntos por grupo, Total N={n_total_plot})", 
                                   fontsize=16, fontweight='bold')
        axes_pca[ax_idx].set_xlabel("PC1 (Regularidad Académica)")
        axes_pca[ax_idx].set_ylabel("PC2 (Promedio Natural)")
        
        # Añadir líneas de cuadrantes de referencia
        axes_pca[ax_idx].axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
        axes_pca[ax_idx].axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
        
        # Limpiar leyenda interna para unificarla
        axes_pca[ax_idx].get_legend().remove()
        
    fig_pca_bal.suptitle("Geometría del Espacio de Trayectorias (Muestras Equitativas)\n"
                         "Disposición Territorial del Talento sin Sesgo de Volumen Poblacional", 
                         fontsize=19, fontweight='bold', y=1.02)
    
    # Leyenda Unificada
    fig_pca_bal.legend(handles=final_legend, title="Grupos Teóricos", 
                       loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.93), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(IMG_DIR / "pca_balanceado_por_carrera.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Gráfica de PCA Balanceado guardada exitosamente.")

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
    
    # ── NUEVO: Distribución de Frecuencias por Perfil y Carrera ───────────────────
    print("\n📊 Generando Histograma de Distribución de Frecuencias...")
    plt.figure(figsize=(12, 7))
    
    # Convertir el perfil a categórico para asegurar un orden visual coherente
    orden_perfil = ["Sin atraso", "Rezago", "Pragmático", "Perfeccionista"]
    df["Perfil_Teorico"] = pd.Categorical(df["Perfil_Teorico"], categories=orden_perfil, ordered=True)
    
    ax_bar = sns.countplot(
        data=df, x="Carrera", hue="Perfil_Teorico", 
        palette=pal_teorica, edgecolor="black", alpha=0.9
    )
    
    # Añadir etiquetas numéricas exactas arriba de cada barra
    for p in ax_bar.patches:
        val = int(p.get_height())
        if val > 0:
            ax_bar.annotate(f'{val}', (p.get_x() + p.get_width() / 2., val),
                           ha='center', va='center', xytext=(0, 8), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
            
    plt.title("Distribución Poblacional de Perfiles Académicos (S4)\n"
              "Validación de Frecuencias Absolutas por Licenciatura", fontsize=16, pad=15, fontweight='bold')
    plt.ylabel("Número Total de Estudiantes")
    plt.xlabel("Licenciatura")
    plt.ylim(0, df["Perfil_Teorico"].value_counts().max() * 0.7) # Ajustar altura para que quepan labels y no sobre aire
    # Pero como está dividido por carrera, el máximo por barra es menor. Usamos un ajuste dinámico:
    max_h = max([p.get_height() for p in ax_bar.patches if p.get_height() > 0])
    plt.ylim(0, max_h * 1.15)
    
    plt.legend(title="Grupo Teórico", loc='upper right', fontsize=11)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "conteo_perfiles_por_carrera.png", dpi=300)
    plt.close()
    print(f"✅ Gráfica de Distribución de Volumen guardada en: {IMG_DIR / 'conteo_perfiles_por_carrera.png'}")
    
    print("\n🎉 Proceso finalizado exitosamente.")

if __name__ == "__main__":
    main()
