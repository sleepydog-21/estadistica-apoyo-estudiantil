# Tesis Estadística: Enlace Simbiótico (Antigravity Schema)

Este documento es el **contrato de intercambio de conocimiento** entre el entorno de desarrollo aplicado de la Tesis (`/Users/sleepydog/Documents/tesis estadistica/`) y la Bóveda Ontológica Central (`/Users/sleepydog/Documents/boveda/`).

**El agente Antigravity en la nube es el responsable de mantener este puente activo en ambas direcciones.**

## 1. Propósito y Regla de Simbiosis (Core Principle)

- **La Tesis** funciona como un "laboratorio aplicativo". Aquí se escribe código (`src/`), se analizan datos y se prueban modelos (Ej. Supervivencia, PCA).
- **La Bóveda** es el repositorio de conocimiento teórico, arquitectónico y metodológico curado.
- **La Simbiosis:** La bóveda debe guiar el código de la tesis con rigor teórico. A su vez, los métodos exitosos descubiertos en la tesis deben nutrir y expandir permanentemente la bóveda.

---

## 2. Flujo de Importación: Usar la Bóveda como Contexto

Antes de proponer metodologías, resolver problemas complejos de machine learning o diseñar arquitecturas de código en este proyecto, el agente debe:

1. **Consultar el Índice Central:** Leer `/Users/sleepydog/Documents/boveda/wiki/index.md`.
2. **Extraer Contexto:** Leer las notas relevantes en `boveda/wiki/concepts/` y `boveda/wiki/methods/`.
3. **Alineación:** Asegurar que las soluciones de código en la tesis se adhieran al rigor y a las definiciones ya establecidas en la bóveda. Priorizar este conocimiento centralizado sobre el conocimiento genérico de internet.

---

## 3. Flujo de Exportación: Reverse Ingestion (Cross-Project Growth)

Cuando la resolución de un problema en la tesis genera un avance significativo (ej. un pipeline estadístico universal, una optimización de RAM, o un nuevo modelo de Análisis de Supervivencia), se **prohíbe** que ese conocimiento se quede aislado aquí.

El agente debe ejecutar una "Ingestión Inversa" hacia la bóveda siguiendo estas reglas estrictas:

1. **Abstracción:** Separar la lógica teórica/metodológica del código *hardcoded* (datos específicos).
2. **Draft-First Workflow:** Generar un artículo denso y riguroso documentando la metodología descubierta y guardarlo en la ruta de borradores de la bóveda:
   `/Users/sleepydog/Documents/boveda/wiki/.drafts/methods/` (o la carpeta correspondiente como `concepts/`).
3. **Protección:** NUNCA modificar directamente una nota publicada en `boveda/wiki/` desde aquí. Si el conocimiento mejora una nota existente, crear un archivo *Merge Proposal* en `.drafts/`.
4. **Registro:** Cada vez que se envíe conocimiento a la bóveda, añadir una entrada al registro en `/Users/sleepydog/Documents/boveda/wiki/log.md`.

---

## 4. Operativa de Código vs Conocimiento

- **Scripts de un solo uso, Análisis de Datos y Logs (`src/`, `Datos/`, etc.):** Pertenecen exclusivamente a la carpeta de la tesis. No inundar la bóveda con código *spaghetti* o visualizaciones de datos transitorias.
- **Patrones de Diseño, Abstracciones y Teoremas:** Pertenecen a la bóveda. Si un script en la tesis es brillante, se documenta su *patrón algorítmico* en la bóveda, pero el archivo ejecutable sigue viviendo en la tesis.
- **Reportes:** Si un cuaderno (Notebook) o script produce un análisis estadístico crítico que deba ser guardado a largo plazo, el agente puede enviar una síntesis de los hallazgos a `/Users/sleepydog/Documents/boveda/wiki/.drafts/reports/`.
