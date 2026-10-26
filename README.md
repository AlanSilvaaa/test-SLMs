# Experimentos con SLM

Implementación de la metodología de `main.tex` de la memoria de título: generación
local de problemas matemáticos para primero a sexto básico con seis SLM y siete
condiciones por modelo. La rúbrica y la skill incluidas siguen siendo provisionales.

Todo se ejecuta en el computador experimental donde se clona este repositorio.
El programa prepara las entradas, inicia **`vllm serve` en `127.0.0.1`**, ejecuta
**`vllm bench serve`** y cierra el servidor. Es el procedimiento de medición de la
memoria. El benchmark oficial utiliza HTTP sobre loopback; no utiliza Internet,
proveedores externos, clientes HTTP propios ni la API Python `LLM.generate`.
No ejecutar entrenamiento, generación ni judge en el computador de desarrollo.

## Prueba rápida con flags

Si necesitas autenticar las descargas de Hugging Face, copiar `.env.example` a
`.env` en la raíz del clon y completar `HF_TOKEN`:

```dotenv
HF_TOKEN=hf_TU_TOKEN
```

El programa carga ese archivo automáticamente antes de preparar el modelo.
Las variables que ya existen en el entorno tienen prioridad. No se realiza un
`login()` persistente: el SDK utiliza
[`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hf_token)
durante la descarga. `.env` está excluido de Git y el token no se incluye en la
configuración, los comandos ni los resultados. Los procesos locales de
serve/bench tampoco reciben `HF_TOKEN`.

En el computador experimental, desde la raíz del repositorio:

```bash
uv run --extra infer main.py -n 5 --model SmolLM3
```

También se aceptan `-model SmolLM3`, `-m SmolLM3` y el ID completo
`HuggingFaceTB/SmolLM3-3B`. No hace falta editar JSON. Se ejecuta solo la condición
base en modo desarrollo, una repetición, con las cinco solicitudes sintéticas
incluidas. Al finalizar se imprime la ruta de `resultados.xlsx`, dentro de
`output/HuggingFaceTB/SmolLM3-3B/<fecha-hora>/`.

Los seis modelos admiten estos alias en `--model`, `-model` o `-m`.
Los alias no distinguen mayúsculas de minúsculas; también se acepta el modelID completo.

| Alias | ModelID |
| --- | --- |
| `SmolLM3` | `HuggingFaceTB/SmolLM3-3B` |
| `llama-1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `granite-2b` | `ibm-granite/granite-3.3-2b-instruct` |
| `phi4-mini` | `microsoft/Phi-4-mini-instruct` |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` |
| `granite-8b` | `ibm-granite/granite-3.3-8b-instruct` |

Por ejemplo, para probar cinco solicitudes con Qwen3 en el equipo experimental:

```bash
uv run --extra infer main.py -n 5 --model qwen3-4b
```

Cada comando ejecuta un único modelo en condición base. Los alias se definen en
[`common/models.py`](common/models.py) y se validan contra
[`configs/models.json`](configs/models.json).

Para ejecutar los seis modelos base, uno después de otro:

```bash
uv run --extra infer main.py -n 5 -models all
```

También se acepta `--models all`. `-n` indica solicitudes **por modelo**: este
ejemplo produce 30 generaciones, con un Excel por modelo. Cada servidor se cierra
antes de iniciar el siguiente. Si una ejecución falla, el comando se detiene y
conserva los resultados anteriores. Se aplican los mismos flags `--requests`,
`--offline`, `--models-dir`, `--output` y `--port` a todos los modelos.
No combinar `--models all` con `--model` ni con `--revision`, porque cada modelo
tiene su propio commit. Para más de cinco solicitudes, proporcionar un archivo de
desarrollo suficiente mediante `--requests` y su cantidad con `-n`.
Este comando mantiene una repetición en desarrollo; no ejecuta las demás
estrategias ni el benchmark final.

Los pesos se buscan primero en **`models/` dentro de la raíz del repositorio**,
independientemente del directorio desde el que se invoque el programa, usando la estructura de caché de
Hugging Face por modelID, por ejemplo:

```text
models/models--HuggingFaceTB--SmolLM3-3B/snapshots/<commit>/
```

Si no hay un snapshot completo, se descarga allí antes de iniciar serve/bench.
La primera preparación requiere conexión y permisos de escritura en `models/`.
Después se utiliza la ruta local y se registra el commit exacto resuelto. Si ya
hay una copia completa, se reutiliza sin consultar la red. Para impedir descargas:

```bash
uv run --extra infer main.py -n 5 --model SmolLM3 --offline
```

Otros flags: `--models-dir /otra/carpeta`, `--revision <commit-exacto>`,
`--requests /ruta/solicitudes.jsonl`, `--output /ruta/resultados` y `--port 8001`.
Si el puerto 8000 está ocupado, identificar el proceso con
`ss -ltnp 'sport = :8000'`. Solo detenerlo si corresponde a una ejecución propia
que ya no se necesita; para conservar otro servicio, elegir un puerto libre con
`--port`. El programa no conecta el benchmark a un servidor ajeno ni lo termina.
`-n` selecciona las primeras N solicitudes de desarrollo; si el archivo tiene
menos, se rechaza la ejecución antes de descargar, sin repetir ni inventar entradas.
Para más de cinco solicitudes hay que proporcionar un archivo suficiente con
`--requests`. La selección queda registrada como `request_limit` en la configuración
de salida. El benchmark final no permite limitar solicitudes mediante ese campo.

La descarga preparatoria usa Hugging Face Hub; la generación y la medición siguen
usando exclusivamente los comandos oficiales locales `vllm serve` y
`vllm bench serve`. En este computador de desarrollo solo se realizan pruebas sin
modelos: la descarga y la ejecución corresponden al hardware experimental.

## Preparación del computador experimental

En la GX10 con Ubuntu y Python 3.12 del sistema, instalar las cabeceras antes
de ejecutar vLLM. Triton compila extensiones que requieren `Python.h`:

```bash
sudo apt update
sudo apt install python3.12-dev
```

Python 3.12 y `uv`. Para el flujo con configuración completa, preparar previamente en ese computador
los pesos, tokenizadores y dependencias. El modo rápido anterior puede descargar
los pesos faltantes automáticamente. Después, trabajar con rutas absolutas locales:

```bash
uv sync --locked --extra infer
# En el equipo de entrenamiento, en un clon separado:
uv sync --locked --extra train
```

El extra `infer` fija vLLM **0.15.1** y pandas, requerido por su dataset `custom`.
En Linux ARM64 selecciona el build oficial **0.15.1+cu130** y PyTorch
**2.9.1+cu130** desde su índice CUDA 13, junto a torchvision y torchaudio.
Esto corrige la instalación PyTorch CPU observada en la GX10. Ver la
[guía de instalación y comprobación de la GX10](docs/gx10-setup.md).
La ejecución completa con GB10 debe comprobarse en el hardware experimental.
Si ese entorno necesita otra distribución, actualizar y validar conjuntamente
comandos, formato de resultados y lockfile; no asumir compatibilidad por cambiar
solo la versión. `uv sync --locked` sin extras instala herramientas ligeras.

El ejecutor activa `HF_HUB_OFFLINE=1` y `TRANSFORMERS_OFFLINE=1` en sus procesos y
el tokenizador se abre con `local_files_only=True`. No activa `trust_remote_code`.
Registrar `model.id` según `configs/models.json`, `model.path` como carpeta local y
`model.revision` como commit exacto de 40 caracteres. Se conserva un fingerprint de
los archivos locales: declarar una revisión no demuestra por sí solo su procedencia.

## Datos y configuración

Para probar únicamente SmolLM3 base con cinco solicitudes y una repetición,
seguir [la guía de prueba técnica](examples/smoke/README.md) y usar
`configs/smollm3-smoke.example.json`. Este flujo de desarrollo no requiere preparar
el dataset ni el benchmark final; si el archivo reservado existe, sí se comprueba
la separación. Las cinco solicitudes incluidas son sintéticas y no representan OA
oficiales. El modo benchmark conserva sus requisitos completos.

Ver [data/README.md](data/README.md). Exportar solo una versión aceptada por candidato:

```bash
uv run main.py import-benchmark \
  --source /ruta/MathGenBench-CL/benchmark.json --output data/benchmark.jsonl
uv run main.py prepare --dataset data/accepted.jsonl --benchmark data/benchmark.jsonl
```

`prepare` exige 40 ejemplos por OA y cinco solicitudes por OA, con 59 OA.
Produce train/validation 34/6 por OA (2006/354 pares), semilla 42, y cuatro
identificadores de demostraciones por OA tomados solo de train. La revisión
pedagógica y la detección de paráfrasis siguen siendo necesarias. Se rechazan IDs
compartidos entre train/validation aunque sus versiones y textos sean distintos.

Completar `configs/experiment.json` a partir del ejemplo. Las rutas de datos son
relativas a la raíz situada sobre `configs/`. Para ejecutar se requieren:

- `benchmark`: las solicitudes finales reservadas.
- `development`: otras solicitudes para ajuste y pruebas, sin IDs ni solicitudes
  literales del benchmark. Desarrollo tampoco puede solaparse con los ejemplos.
- `oa_registry`: el registro local revisado de los 59 OA, con curso, eje y texto
  exacto. El benchmark se contrasta con su identidad y con la distribución de la
  tabla de la memoria, además de exigir 295 solicitudes.
- `phase`: `development` utiliza las solicitudes de desarrollo; `benchmark` exige
  el conjunto completo y al menos dos repeticiones. El ejemplo usa tres repeticiones:
  es una decisión operativa explícita, no un número fijado en la memoria.
- `load_conditions`: descripción de carga del equipo para las mediciones finales.
  `container_image`: identidad y digest del contenedor, obligatoria al entrenar.

Temperatura 0.6, top_p 0.95, salida máxima 1024 y memoria reservable 0.85 son valores
iniciales del proyecto. La memoria no fija esos números: deben ajustarse en
**desarrollo**, documentarse y mantenerse comunes antes del benchmark. Contar las
entradas de las siete condiciones con los seis tokenizadores antes de fijar el
contexto común; el límite inicial es 4096, incluyendo reserva de salida. El
preflight rechaza excesos y verifica que texto y tokens de la plantilla coincidan.

## Generación y medición local

```bash
uv run --extra infer main.py run --module base_model --action infer --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 1 --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 2 --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 3 --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 4 --config configs/experiment.json
uv run --extra infer main.py run --module skill --action infer --config configs/experiment.json
uv run --extra infer main.py run --module lora --action infer --config configs/lora-infer.json
```

El mismo comando prepara prompts con la plantilla del modelo, inicia el servidor
local y ejecuta una invocación de `vllm bench serve` por repetición. Se utiliza
`--dataset-name custom`, `--skip-chat-template`, `--disable-shuffle`,
`--no-oversample`, `--max-concurrency 1`, `--save-result` y `--save-detailed`.
La plantilla ya aplicada no se aplica nuevamente. Se respetan EOS y los valores
explícitos de decodificación. Los ejemplos A/B/C/D rotan según la posición de la
solicitud dentro del OA, con la misma asignación entre modelos.

Las generaciones se guardan en `output/<modelID>/<fecha-hora>/`: abrir
**`resultados.xlsx`** para revisar enunciados, soluciones, errores y métricas.
La fecha usa la hora de Chile (`America/Santiago`). Los archivos técnicos quedan
en `detalles/`. Ver [la guía de resultados](docs/results.md), incluida la conversión
a Excel de ejecuciones anteriores.

Cada ejecución conserva configuración, código, versiones,
checksums, plantilla efectiva, prompts y tokens, comandos exactos, logs de serve y
bench, resultados nativos, generaciones y resúmenes. No se sobrescriben ejecuciones.
Una interrupción conserva logs y repeticiones finalizadas; vLLM guarda el resultado
nativo al terminar cada invocación, por lo que no se garantiza recuperar la
repetición que estaba en curso. `status.json` aparece solo al completar el flujo.

La salida del generador se solicita como JSON mediante el prompt común, sin
forzar una gramática. Se preservan errores de parseo aparte de las valoraciones.
Si el JSON contiene un enunciado y una solución vacía/ausente, el enunciado puede
evaluarse y la ausencia de solución se registra separadamente. Si se alcanza el
límite de salida se registra explícitamente `output_limit_reached`; vLLM 0.15.1 no
conserva `finish_reason` en el resultado detallado. Revisar estos casos antes de
cerrar el experimento. No se inventan motivos de terminación.

Para LoRA, configurar `adapter` y `adapter_sha256` del adaptador seleccionado.
Se registra como módulo local `math`, utilizado también en la solicitud inicial
del benchmark; no se sortean adaptadores. Se verifica su checksum, identificador
y revisión del modelo base. Puede cambiar la ruta de pesos entre computadores.
La condición LoRA utiliza solo la instrucción común.

La skill debe pasar revisión pedagógica y pruebas de desarrollo con los modelos.
En benchmark, `skill_review` debe señalar un registro local con `status: "approved"`,
`skill_sha256` del archivo revisado, `pedagogical_review` y `development_runs`.
Es evidencia que debe completar el investigador; no se incluye una aprobación ficticia.

Ver definiciones, alcance de calentamiento y limitaciones en
[docs/performance.md](docs/performance.md).

## Entrenamiento y selección LoRA

Solo en el equipo de entrenamiento. Crear tres configuraciones por modelo,
cambiando únicamente `learning_rate`: 0.00005, 0.0001 y 0.0002.

```bash
uv run --extra train main.py run --module lora --action train --config configs/lora-lr1.json
uv run --extra train main.py run --module lora --action train --config configs/lora-lr2.json
uv run --extra train main.py run --module lora --action train --config configs/lora-lr3.json
uv run main.py select-lora output/RUN1 output/RUN2 output/RUN3 --output selected-adapter.json
```

Son 18 entrenamientos independientes. Rango 16, alpha 32, dropout 0.05, cinco épocas,
BF16, microbatch 1, acumulación 16, AdamW no fused (0.9/0.999/1e-8), cosine hasta
cero, warmup 10 %, weight decay 0, clipping 1 y gradient checkpointing. Sin packing,
cuantización ni early stopping. Se conserva el último grupo incompleto. Pérdida
solo sobre respuesta del asistente, excluyendo solicitud y padding.

Se requiere semilla 42 y `all-linear`; se guardan los nombres de parámetros
adaptados. Verificar en el equipo de entrenamiento que la política cubra atención
y MLP, excluyendo biases, embeddings y salida en cada arquitectura. El código usa
Transformers Trainer con máscaras explícitas; **TRL no se utiliza**. Se registran
las advertencias Python del entrenamiento y las versiones del entorno.

Seleccionar por menor pérdida de validación; desempatar por época más temprana y
learning rate menor. No comparar pérdidas entre tokenizadores diferentes ni
reentrenar con validación. La prueba de memoria/compatibilidad con el modelo mayor
y secuencias más largas debe realizarse antes de los 18 entrenamientos en Patagón.

## Judge y revisión docente

La rúbrica original se conserva. `configs/rubric.v0.2-draft.json` incorpora
explícitamente la regla de la memoria sobre errores lingüísticos menores. Sigue
siendo **draft**: revisar, calibrar y versionar antes de aprobarla. El ejecutor
rechaza rúbricas draft y exige exactamente las cinco dimensiones previstas.

Completar `configs/judge.json` con Gemma 3 27B IT. El juez usa los mismos comandos
oficiales locales, después de medir los generadores. En `phase: "development"`
permite calibrar con generaciones de desarrollo y una rúbrica aprobada para esa
revisión. Para el benchmark requiere `judge_review`: `status: "approved"`,
`rubric_sha256`, `manual_review` y `development_sources` (rutas absolutas a archivos
de generaciones de desarrollo), separados de las solicitudes finales. La fase de
las generaciones debe coincidir con la del juez.

```bash
uv run --extra infer main.py judge --config configs/judge.json \
  --source output/MODEL/FECHA/detalles/generations.jsonl --rubric configs/rubric.v1.json
uv run main.py summarize --source output/JUDGE/evaluations.jsonl --output summary.json
```

Dos etapas: (1) enunciado, curso/OA y rúbrica, sin solución ni identidad del
modelo; (2) solo si es resoluble, comprobar la solución y registrar procedimiento
independiente y corrección. La segunda etapa ve la propuesta, como indica la memoria;
no equivale a una resolución ciega previa. En el dataset, la resolución independiente
antes de consultar la propuesta sigue siendo responsabilidad del investigador.
Los puntajes de solución y ajuste a la solicitud quedan fuera del conjunto de cinco.
Los errores técnicos se conservan como ausentes y no se convierten en puntajes 0.

```bash
# Muestra del dataset: 118 problemas, dos por OA
uv run main.py sample-dataset --train data/train.jsonl --validation data/validation.jsonl \
  --demonstrations data/demonstrations.json --output output/revision-dataset
# Muestra final: suministrar las ejecuciones de modelos y estrategias a comparar
uv run main.py export-human --source output/MODEL/FECHA1/detalles/generations.jsonl \
  --source output/MODEL/FECHA2/detalles/generations.jsonl --output output/revision-docente
```

La muestra final usa semilla 42, mínimo 30 problemas por modelo y cubre OA, cursos
y estrategias presentes en las fuentes; puede superar 30 para dar esa cobertura.
Se baraja antes de exportar formularios separados de enunciado y solución. El mapa
de identidad y los archivos privados no deben entregarse al docente. `--all`
exporta todas las filas sin muestreo. Proporcionar todas las condiciones necesarias:
no se inventa cobertura de ejecuciones que no se entregaron.

Para comparar anotaciones independientes:

```bash
uv run main.py agreement --judge output/JUDGE1/evaluations.jsonl \
  --judge output/JUDGE2/evaluations.jsonl --human data/human.jsonl \
  --mapping output/revision-docente/private_mapping.csv --output output/acuerdo.json
```

Formato de la anotación humana JSONL: `blind_id`, `evaluador`, `rubric_version`,
`statement` con los cinco IDs y objetos `{score, reason}`, y `solution` con
`{score, reason}` o null si no se evaluó. Justificar solo 0. Se verifica la misma
versión de rúbrica, se calcula acuerdo por dimensión y conjunto, y acuerdo de
solución solo entre resolubles por ambos evaluadores. Se exportan discrepancias
por modelo/estrategia, sin modificar puntuaciones originales. Su resolución
fundamentada debe conservarse en un registro nuevo por el investigador.

## Verificación

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Las pruebas usan datos sintéticos y procesos simulados, sin pesos, modelos ni GPU.
La aceptación pedagógica, revisiones docentes y compatibilidad del hardware se
comprueban fuera de estas pruebas. El informe histórico de auditoría se conserva
en `docs/`; las correcciones posteriores se documentan por separado.
