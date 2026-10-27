# Tanda piloto nocturna

Genera las solicitudes de la hoja «Benchmark piloto» de
`dataset_y_benchmark_piloto_v1.xlsx`: 59 OA, cinco solicitudes por OA, 295 por modelo.
Ejecuta los seis generadores de `configs/models.json`, en condición base y en
secuencia. Una pasada da 1770 generaciones solicitadas. No ejecuta judge ni
entrenamiento y no repite las mediciones. No permite limitar el conjunto.

Esta es una fase explícita `pilot`, distinta de desarrollo y benchmark final.
Conserva el estado `draft`: siguen pendientes la revisión curricular y la búsqueda
de similitudes indicadas en la guía del Excel. Las generaciones de estas
solicitudes reservadas no deben utilizarse como entrenamiento, demostraciones ni
para ajustar instrucciones. El benchmark final conserva sus requisitos de
registro aprobado, separación y mediciones repetidas.

## Preparación una sola vez

En este computador ya se preparó `data/pilot/` desde los dos Excel de la memoria,
y se empaquetó como `data/piloto-noche.zip`. Los datos están excluidos de Git:
actualizar el repositorio no los copia al computador experimental.

Para copiarlos al GX10, desde este computador:

```bash
scp data/piloto-noche.zip fondef@gx10-01:~/alan/test-SLMs/data/
```

En el GX10, actualizar el código y, desde la raíz del repositorio:

```bash
uv sync --locked --extra infer
unzip -n data/piloto-noche.zip -d data
```

El paquete debe quedar como `data/pilot/manifest.json`, `oa.json`,
`benchmark.jsonl` y `dataset-requests.jsonl`. No sobrescribir una versión anterior;
para otra preparación, usar otra carpeta y señalarla con `--data`.

Para reconstruir el paquete desde los originales sin modelos ni GPU:

```bash
uv run --no-sync main.py prepare-pilot \
  --source /ruta/dataset_y_benchmark_piloto_v1.xlsx \
  --oa-source /ruta/used_oa_and_benchmark.xlsx \
  --output data/pilot
```

La preparación conserva las solicitudes, normaliza el curso y obtiene el texto y
eje del OA del registro curricular. Verifica los 59 OA, su distribución por
curso/eje, cinco solicitudes por OA y separación literal respecto de los
candidatos del dataset. No importa sus soluciones ni los acepta para entrenar.
Conserva hashes de los Excel originales y de los archivos preparados. Estas
verificaciones no sustituyen revisión pedagógica ni detección de paráfrasis.

## Un comando para dejarlo trabajando

En el GX10:

```bash
uv run --no-sync main.py pilot --background
```

Imprime el PID y un archivo de log con el comando `tail -f` para ver el avance.
El proceso se separa de la sesión SSH y puede continuar tras cerrarla; el equipo
debe permanecer encendido y sin suspensión. Las descargas iniciales se hacen en
`models/`, usando `HF_TOKEN` de `.env` cuando corresponde. Con `--offline` se
prohíben descargas. También admite `--models-dir`, `--port`, `--output` y `--data`.
Sin `--background`, permanece conectado a la terminal.

Cada modelo usa los mismos parámetros iniciales del proyecto: semilla 42,
temperatura 0.6, top_p 0.95, contexto 4096 y salida máxima 1024 tokens. SmolLM3
conserva `enable_thinking: false`. La plantilla y los parámetros efectivos se
registran por ejecución. Los valores no certifican calidad ni compatibilidad GPU;
la ejecución real solo se comprueba en el hardware experimental.

## Resultados y fallos

- Cada modelo guarda `output/<modelID>/<fecha-hora>/resultados.xlsx` y `detalles/`.
- La tanda guarda `output/tandas-piloto/<fecha-hora>/status.json`, con modelo en
  curso, ejecuciones completadas, rutas de Excel y fallos. El log de lanzamiento
  está junto a esas carpetas y conserva los mensajes de avance.
- Si falla un modelo, registra su traceback y continúa con el siguiente. Si hay
  fallos, termina con código distinto de cero y estado `completed_with_errors`.
- Una interrupción del proceso coordinador detiene la secuencia y conserva su
  estado; el cierre normal o por interrupción del generador cierra vLLM. Las
  repeticiones terminadas se conservan; no se garantiza recuperar una invocación
  de bench interrumpida. No hay reanudación automática: otra invocación crea una
  tanda nueva con los seis modelos y conserva la anterior.
- El servidor solo escucha en `127.0.0.1`. Generación y mediciones usan
  `vllm serve` y `vllm bench serve`. El consumo y el tiempo total dependen del GX10;
  no se garantiza que la tanda complete dentro de una noche.

«Completa» indica que terminó el flujo técnico, no que todos los problemas sean
correctos. Los errores de formato, límites de salida y soluciones ausentes siguen
registrados en cada Excel. El judge se ejecuta posteriormente, por separado.
