# Experimentos con SLM

Comparación de estrategias para generar problemas matemáticos verbales de educación
básica chilena, según la metodología de la memoria y los proyectos LoRA y
MathGenBench-CL. La rúbrica y la skill incluidas son provisionales.

El flujo consiste en clonar este repositorio en el computador experimental,
preparar allí las dependencias y los pesos, y ejecutar los comandos localmente.
La inferencia y el judge usan la API Python de vLLM. No hay servicios HTTP,
URLs de conexión, puertos ni un computador remoto que ejecute los modelos.

## Organización

| Carpeta | Responsabilidad |
| --- | --- |
| `base_model/` | Instrucción común, sin adaptación |
| `few_shot/` | 1 a 4 demostraciones del mismo OA y rotación cíclica |
| `skill/` | Carga determinista de las instrucciones de `SKILL.md` |
| `lora/` | Entrenamiento, selección del adaptador e inferencia |
| `common/` | Contratos, datos, prompts, ejecución local de vLLM y métricas |
| `judge/` | Evaluación ciega, reportes y formularios de revisión docente |
| `configs/` | Configuraciones y rúbrica versionada |

El nombre de la carpeta Python es `few_shot`, su flag es `--module few-shot`.
Solo LoRA requiere entrenamiento.

## Preparación del computador experimental

Python 3.12 y `uv`. Desde la raíz del repositorio clonado:

```bash
# Inferencia y judge locales
uv sync --locked --extra infer
uv run main.py --help
# Entrenamiento LoRA, en el computador destinado a ello
uv sync --locked --extra train
```

Se recomienda usar clones separados para entrenamiento e inferencia si se realizan
en equipos distintos. `uv sync --locked` sin extras instala solo herramientas
ligeras y de desarrollo. No instala vLLM ni PyTorch.

El extra `infer` fija vLLM mediante el lockfile. Comprobar en el equipo experimental
su compatibilidad con CUDA y la arquitectura. Para GB10/ARM64 puede requerirse una
imagen o wheel compatible. Si se ajustan las dependencias para ese entorno,
hacerlo con `uv` y conservar su lockfile y las versiones efectivas.

Descargar previamente los pesos, el tokenizador y las dependencias en ese mismo
computador. Completar `model.path` con la ruta absoluta local y `model.revision`
con el commit exacto del modelo. Antes de las pruebas offline, exportar
`HF_HUB_OFFLINE=1` y `TRANSFORMERS_OFFLINE=1`. No se activa `trust_remote_code`
automáticamente ni se almacenan credenciales en el repositorio.

## Datos y configuración

Los contratos están descritos en [data/README.md](data/README.md).
Para reutilizar las solicitudes originales de MathGenBench-CL:

```bash
uv run main.py import-benchmark \
  --source /ruta/MathGenBench-CL/benchmark.json --output data/benchmark.jsonl
uv run main.py prepare --dataset data/accepted.jsonl --benchmark data/benchmark.jsonl
```

`prepare` requiere 40 ejemplos aceptados por cada uno de los 59 OA y cinco
solicitudes de benchmark por OA. Genera 2006 ejemplos de train y 354 de validación
(34/6 por OA, semilla 42), más cuatro demostraciones por OA tomadas solo de train.
Prioriza habilidades distintas. Revisar la selección antes de fijarla y conservar
los mismos splits para todos los modelos y learning rates. La importación del
benchmark no sustituye su validación ni importa evaluaciones antiguas.

Copiar `configs/experiment.example.json` a `configs/experiment.json` y completar
las rutas de pesos y datos. Los archivos de datos se resuelven desde la raíz
situada sobre `configs/`. Los modelos del informe están en `configs/models.json`.
Ajustar `chat_template_kwargs` para cada familia antes de fijar el experimento.

## Inferencia local

```bash
uv run --extra infer main.py run --module base_model --action infer --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 1 --config configs/experiment.json
uv run --extra infer main.py run --module few-shot --action infer --shots 4 --config configs/experiment.json
uv run --extra infer main.py run --module skill --action infer --config configs/experiment.json
uv run --extra infer main.py run --module lora --action infer --config configs/lora-infer.json
```

Repetir few-shot con `--shots 2` y `3`. Los ejemplos A, B, C, D rotan cíclicamente
según la posición de la solicitud dentro de su OA, manteniendo la asignación entre
modelos. vLLM se carga una vez por comando y procesa las solicitudes secuencialmente.
El límite inicial de 4096 tokens incluye entrada y salida reservada. Todas las
entradas se comprueban antes de generar y no se truncan.

Para inferencia LoRA, configurar `adapter` y `adapter_sha256` con el adaptador
seleccionado. Se carga directamente con `LoRARequest`, sin fusionarlo con el modelo
base. Su revisión debe coincidir con `provenance.json`. La ruta de los pesos puede
ser distinta entre el computador de entrenamiento y el de inferencia.
La condición LoRA utiliza únicamente la instrucción común.

Cada ejecución crea una carpeta en `output/` con configuración, versiones,
checksums, solicitudes, respuestas originales, errores y métricas. Se guarda cada
respuesta al terminar su solicitud. Una ejecución interrumpida conserva lo escrito
y no tiene `status.json` de finalización. `repetitions` repite el conjunto con
semillas `seed + repetición`. `gpu_memory_utilization` controla la fracción de
memoria que vLLM puede reservar, con valor inicial 0.85.

## Entrenamiento y selección LoRA

Crear tres configuraciones por modelo, cambiando únicamente `learning_rate`:
`0.00005`, `0.0001` y `0.0002`. Ejecutar solo en el equipo de entrenamiento:

```bash
uv run --extra train main.py run --module lora --action train --config configs/lora-lr1.json
uv run --extra train main.py run --module lora --action train --config configs/lora-lr2.json
uv run --extra train main.py run --module lora --action train --config configs/lora-lr3.json
uv run main.py select-lora output/RUN1 output/RUN2 output/RUN3 --output selected-adapter.json
```

Son 18 entrenamientos para los seis modelos. Cada orden inicia un adaptador nuevo.
Se fija rango 16, alpha 32, dropout 0.05, cinco épocas, BF16, microbatch 1,
acumulación 16, AdamW, cosine, warmup 10 % y gradient checkpointing. Sin packing
ni cuantización del modelo base. La pérdida excluye tokens de la solicitud y
padding. Se verifican las máscaras y se rechazan secuencias demasiado largas.
`all-linear` de PEFT adapta proyecciones lineales y excluye la capa de salida.
Revisar la lista registrada de parámetros adaptados para cada arquitectura.

Se conserva el checkpoint con menor pérdida de validación. Entre las tres tasas
se selecciona menor pérdida, luego época más temprana y luego menor tasa. No se
comparan pérdidas de modelos diferentes ni se reentrena con validación.

## Judge reutilizable

Tras confirmar la rúbrica, copiar `configs/rubric.draft.json` a una versión nueva,
revisar descriptores y ejemplos, asignar versión y marcar `status: "approved"`.
El código rechaza rúbricas draft. Calibrar con problemas de desarrollo y revisión
manual antes del benchmark. Completar `configs/judge.json` desde el ejemplo para
cargar Gemma 3 27B IT localmente, después de medir los generadores.

```bash
uv run --extra infer main.py judge --config configs/judge.json \
  --source output/RUN/generations.jsonl --rubric configs/rubric.v1.json
uv run main.py summarize --source output/JUDGE/evaluations.jsonl --output summary.json
uv run main.py export-human --source output/RUN/generations.jsonl --output output/revision-docente
```

Primero se puntúa únicamente el enunciado, sin solución ni identidad del generador.
El ajuste adicional a la solicitud se registra aparte. Si el enunciado es resoluble,
se obtiene una solución independiente antes de mostrar la propuesta y luego se
contrasta esta última. Se conservan instrucciones y respuestas de todas las etapas.
El cumplimiento conjunto incluye solo las cinco dimensiones del enunciado.

Los reportes mantienen todas las generaciones en el denominador y muestran los
juicios ausentes por separado. La corrección se informa entre casos resolubles.
Los formularios humanos separan enunciado, solución y mapa privado. Corresponde al
investigador seleccionar la muestra estratificada prevista en el informe y
conservar revisiones y discrepancias sin sobrescribir las puntuaciones originales.

## Verificación y mediciones

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Las pruebas usan datos y módulos simulados. No cargan modelos ni ejecutan
entrenamiento o evaluación real. La compatibilidad GPU y las plantillas reales
deben comprobarse en el equipo experimental.

La inferencia registra tokens, tiempo de la llamada local, RAM y swap. TTFT y
velocidad de decode solo se registran si vLLM proporciona sus tiempos por solicitud.
Ver las definiciones y el ajuste respecto de la medición con `bench serve` del
informe en [docs/performance.md](docs/performance.md).

Referencias: [vLLM Python API](https://docs.vllm.ai/en/stable/api/vllm/),
[LoRA en vLLM](https://docs.vllm.ai/en/stable/features/lora/),
[PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora).
