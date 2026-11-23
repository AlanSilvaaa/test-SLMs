# Judge en Patagón

La preparación descarga Gemma 3 27B IT en un nodo CPU y fija el commit del
modelo. Se requiere acceso autorizado al repositorio de Gemma en Hugging Face
mediante la credencial instalada en Patagón. El trabajo de evaluación monta el
directorio del proyecto y usa los pesos locales sin conexión. Por defecto
solicita dos GPU de la partición AI para paralelismo tensorial y emplea un
contenedor vLLM 0.15.1 mediante Pyxis. Se puede indicar una imagen ya disponible
con CONTAINER_IMAGE y elegir el nombre de Pyxis con CONTAINER_NAME. Registrar la imagen
en container_image de la configuración.
La [guía de Slurm de Patagón](https://patagon.uach.cl/tutoriales/how-to-launch-slurm-jobs)
documenta la ejecución con Pyxis.

Preparar primero la configuración a partir de configs/judge.example.json.
Indicar el snapshot exacto como model.path y model.revision, y fijar
tensor_parallel_size en 2 con el lanzador de dos GPUs. Mantener rutas relativas
a la raíz del proyecto o absolutas dentro de su directorio montado. Verificar
que cada configuración tenga la fase de su fuente.

~~~bash
# Preparación de pesos en un nodo CPU
sbatch scripts/patagon/prepare-judge.slurm COMMIT_DE_GEMMA

# Después de la preparación, ensayo de desarrollo con la rúbrica v0.3
sbatch judge.slurm configs/judge.patagon-development.json \
  examples/judge/calibration.jsonl configs/rubric.v0.3-draft.json --allow-draft

# Corrida candidata completa, con una fila de salida por generación de origen
sbatch judge.slurm configs/judge.patagon-candidate.json \
  data/generations-judge.jsonl configs/rubric.v0.3-draft.json --allow-draft
~~~

Los ocho casos de desarrollo son ejemplos sintéticos separados de las
generaciones del benchmark. El archivo expected-observations.json enumera
resultados esperados para comprobaciones puntuales. La calibración final
requiere cotejar los juicios con una revisión manual y fijar la rúbrica.
En la corrida candidata quedan los fallos de formato como filas sin puntuar.
Una ejecución final exige generaciones de fase benchmark y la revisión
documentada de la rúbrica y del juez.

Slurm guarda las salidas del trabajo en slurm-tesis-judge-ID.out y
slurm-tesis-judge-ID.err. El segundo archivo muestra el avance del judge por lotes. El trabajo conserva el JSONL de
evaluaciones, sus mensajes, resultados originales del judge, el resumen y
las huellas de los insumos. Si se interrumpe una evaluación, quedan los lotes
completados dentro de la carpeta de esa ejecución.
