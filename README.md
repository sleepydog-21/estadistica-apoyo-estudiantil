# Estudio de tendencias académicas mediante estadística inferencial

Este proyecto tiene como objetivo preparar y analizar datos provenientes del **Vector Excalibur** para inferir el semestre en que los alumnos concluyen su licenciatura, empleando indicadores académicos por semestre. El enfoque principal del análisis es sobre las licenciaturas de **Matemáticas** y **Física**.

## Metodología

La limpieza consistió en cargar las cuatro hojas del Vector Excalibur por licenciatura, eliminar estudiantes con periodos de baja temporal o definitiva, conservar la variable `Generación` como variable de cohorte, construir la variable objetivo `semestre_termino` a partir de *Regularidad gregoriana*, recortar la información a los primeros ocho semestres, renombrar las variables semestrales por dimensión y unir las cuatro fuentes mediante *inner join* por `Cuenta`. 

La variable `semestre_termino` se definió como el primer semestre $k \geq 8$ en el que la regularidad toma valor 1; si no existe tal semestre, se asigna el valor 20. 

## Estructura del Proyecto

* `Datos/`: Contiene los archivos crudos originales y los procesados (ignorados en git).
* `src/notebooks/`: Scripts de análisis y preparación de datos.
* `escrito/`: Documentación formal (en LaTeX) para el trabajo de grado.
