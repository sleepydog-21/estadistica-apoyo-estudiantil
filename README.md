# Estudio de trayectorias académicas mediante estadística inferencial

Este proyecto analiza las trayectorias de egreso en las licenciaturas de **Matemáticas** y **Física** de la Facultad de Ciencias, UNAM. Utilizando datos del **Vector Excalibur**, empleamos técnicas de reducción de dimensionalidad y análisis exploratorio avanzado para identificar patrones de rezago académico.

## Estado Actual del Proyecto

### 1. Metodología de Datos
*   **Tiempo Activo**: Implementación de una corrección de semestres para alumnos con bajas temporales, centrando el análisis en el tiempo real de permanencia.
*   **Variable Objetivo**: Definición de `semestre_termino` basada en el cumplimiento del 100% de créditos (regularidad = 1), con tratamiento de censura en el semestre 20 para casos no concluidos.
*   **Pipeline de Limpieza**: Scripts automatizados que integran Regularidad, Esfuerzo, Promedio Natural y Promedio Real de los primeros 8 semestres.

### 2. Hallazgos del Análisis Exploratorio (EDA)
*   **Divergencia Temprana**: Identificación de una separación estadísticamente significativa (IC 95%) entre alumnos con egreso exitoso y rezago severo desde el segundo año de la carrera.
*   **Paradox de la Censura**: Análisis del sesgo informativo en las medias generacionales recientes (2017-2019), diferenciando entre el sesgo superior (imputación k=20) y el sesgo inferior (solo titulados rápidos).
*   **Predicción Temprana**: Validación de que el Semestre 4 ya contiene suficiente señal (correlación ~-0.6) para predecir si un alumno superará el umbral de los 12 semestres de egreso.

### 3. Reducción de Dimensionalidad (PCA)
*   **Síntesis de Información**: Reducción de 32 variables a los primeros 5 componentes principales, capturando aproximadamente el **90% de la varianza**.
*   **Anatomía del PC1 y PC2**: Identificación del PC1 como factor de rendimiento global y del PC2 como factor de evolución temporal (contraste S1-S4 vs S5-S8).
*   **Recuperación de Información**: Validación de que el espacio latente del PCA separa nítidamente los perfiles de egreso.

### 4. Documentación Formal
*   Tesis modularizada en LaTeX con el **Capítulo de Análisis Exploratorio** finalizado e integrado con visualizaciones optimizadas.

---

---

## Hoja de Ruta (Roadmap)

A continuación se detallan los pasos para completar el análisis y la redacción de la tesina:

### 1. Análisis del Tiempo de Egreso (Supervivencia)
*   **Curvas de Kaplan-Meier**: Ver la probabilidad real de graduarse mes a mes y comparar Matemáticas con Física.
*   **Impacto del Inicio**: Analizar cómo influye el rendimiento del primer año en el tiempo final de carrera.
*   **Modelos de Cox**: Identificar qué métricas (Esfuerzo, Promedio, etc.) son las que más aceleran o retrasan la graduación.

### 2. Exploración de Patrones y Dinámicas
*   **Análisis de Volatilidad**: Ver si los alumnos con notas muy inestables tienen más riesgo de rezago.
*   **Agrupamientos (Clustering)**: Buscar perfiles de alumnos (ej. mucho esfuerzo pero notas medias) para entender mejor la diversidad de trayectorias.
*   **Inercia Académica**: Medir qué tanto un semestre predice al siguiente para localizar puntos críticos de intervención.

### 3. Herramienta de Detección Temprana (Modelado)
*   **Predicción de Rezago**: Usar los componentes del PCA para intentar predecir quién superará el umbral de los 12 semestres.
*   **Validación**: Evaluar qué tan confiable es esta predicción usando solo los datos de la mitad de la carrera.

### 4. Escritura y Cierre
*   **Capítulos Finales**: Redactar la introducción, el marco teórico y las conclusiones.
*   **Detalles Finales**: Armar el resumen, los agradecimientos y dejar el documento listo para su entrega.

---

## Estructura del Repositorio
* `Datos/`: Datos procesados y limpios por carrera.
* `src/notebooks/`: Scripts de procesamiento (1-24) y futuros cuadernos.
* `imagenes/`: Acervo visual optimizado para la tesis.
* `escrito/`: Código fuente de la tesis (LaTeX).
