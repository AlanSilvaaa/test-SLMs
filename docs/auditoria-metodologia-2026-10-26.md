# Contraste del proyecto con la memoria

Revisión del 26 de octubre de 2026. Fuente de verdad:
`/home/alan/Documents/memoria-de-titulo/main.tex`.
SHA-256: `36314291255898d3e89cb4c3f1de730154693913a761d8dc4ab3cf3a3b3b40a0`.
Código revisado sobre HEAD `b2929f4fb01c3c5b330b7beb760a96a36a6f89d8`.
Las líneas citadas corresponden a esa versión de los archivos.

El proyecto coincide con los hiperparámetros centrales de entrenamiento y con
la estructura general del experimento, pero todavía no implementa ni garantiza
todo el protocolo. Esta revisión no cambia parámetros, descriptores, aprobaciones
ni la memoria. Solo añade este informe.

## Parámetros y decisiones que coinciden

| Aspecto | Memoria | Implementación y resultado |
| --- | --- | --- |
| Generadores | Seis modelos, tabla de líneas 688–693 | `configs/models.json`: Llama 3.2 1B Instruct, Granite 3.3 2B Instruct, SmolLM3 3B, Phi-4-mini-instruct, Qwen3 4B Instruct 2507 y Granite 3.3 8B Instruct. Coinciden. |
| Juez | Gemma 3 27B IT, línea 916 | Catálogo y `configs/judge.example.json` coinciden. Son ejemplos de configuración; no hay revisiones reales fijadas. |
| Dataset | 59 OA × 40 pares = 2360, línea 706 | `common/data.py:31`: exige estos tamaños al preparar datos. |
| Benchmark | Cinco solicitudes por OA, 295 en total, línea 910 | `prepare` exige esos tamaños y los mismos OA del dataset. Véase la limitación de validación al ejecutar más abajo. |
| Split | 34/6 por OA, 2006/354, semilla 42, línea 812 | `common/data.py:45`: partición aleatoria reproducible y archivos separados. |
| Few-shot | 1, 2, 3 y 4 ejemplos del mismo OA, solo train, rotación A/B/C/D, líneas 769–771 | `few_shot/infer.py` y `common/generation.py:21`: coincide. La selección favorece habilidades con texto diferente; la diversidad pedagógica requiere revisión. |
| Condiciones | Referencia, cuatro few-shot, skill y LoRA | Están las siete condiciones por modelo; el CLI las ejecuta individualmente. |
| LoRA | r=16, alpha=32, escala 2, dropout=0.05, líneas 820 y 833–835 | `lora/train.py:125`: coincide. Bias desactivado e inicialización estándar explícita. |
| Tasas candidatas | 5e-5, 1e-4 y 2e-4; 18 entrenamientos, líneas 849–851 | `lora/train.py:218` exige tres tasas al seleccionar; el README indica ejecutar tres por modelo. |
| Selección | Menor pérdida; empate por época temprana y tasa menor, línea 851 | `lora/train.py:200` y `:236`: coincide. Se comprueba igualdad de configuración y archivos de entrada entre las tres ejecuciones. |
| Épocas y batch | Cinco épocas, microbatch 1, acumulación 16, conservar último grupo, líneas 853–883 | `TrainingArguments` coincide: cinco épocas, 1/16 y `dataloader_drop_last=False`. No se configura early stopping. |
| Precisión y optimización | BF16 sin cuantización, AdamW no fused, beta1=.9, beta2=.999, epsilon=1e-8 | `lora/train.py:115` y `:153`: coincide. |
| Scheduler | Cosine hasta cero, warmup 10 %, weight decay 0, clipping 1 | `lora/train.py:165`: coincide con la tabla de entrenamiento. |
| Memoria de entrenamiento | Gradient checkpointing, sin packing, padding dinámico | `lora/train.py:41` y `:169`: coincide. |
| Pérdida | Solo respuesta del asistente, excluir solicitud y padding, línea 890 | `encode_example` y `ResponseCollator`: etiquetas -100 para solicitud y padding; comprueba que la plantilla permita aislar la respuesta. Falta validación con tokenizadores reales. |
| Contexto inicial | 4096 incluyendo entrada y salida, línea 700 | Defaults y preflight coinciden. Se rechaza el exceso; no se trunca. La longitud común definitiva debe fijarse antes del experimento. |
| Inferencia LoRA | Solo instrucción común, línea 896 | `lora/infer.py` coincide. |
| Rúbrica | Cinco dimensiones binarias; justificar solo 0 | Los cinco IDs y descriptores draft corresponden en términos generales a la tabla de la memoria. `parse_verdicts` exige justificación para 0 y vacía para 1. |
| Evaluación ciega | Primero enunciado, después solución si es resoluble | `judge/evaluate.py:51` y `:143`: omite identidad y solución propuesta en la primera etapa. |
| Indicadores | Conjunto de cinco dimensiones; corrección aparte entre resolubles, línea 926 | `joint_pass` excluye corrección y ajuste adicional; el denominador de corrección usa resolubles. Véanse los casos especiales más abajo. |
| Trazabilidad | Solicitudes, respuestas, configuración, versiones | Se guardan mensajes, respuestas originales, configuración, checksums y versiones principales en carpetas nuevas. Hay faltantes descritos más abajo. |

## Discrepancias y riesgos que deben resolverse

### 1. La medición de rendimiento difiere del procedimiento escrito

**Prioridad alta.** `main.tex:904` prescribe `vllm bench serve`, TTFT,
tiempo por token, tasa agregada y latencia hasta el último token.
`common/inference.py:88` mide el tiempo de pared de `LLM.generate` y obtiene TTFT
y decode únicamente cuando el motor entrega los tiempos. Esos campos pueden
quedar en null. No existe un resumen de tasa agregada, tiempo por token ni de
medida central y dispersión. `repetitions` vale 1 por defecto, aunque
`main.tex:906` exige mediciones repetidas.

La desviación ya está reconocida en `docs/performance.md`. Además, `AGENTS.md`
exige API Python local y prohíbe clientes HTTP y servidores: hay un conflicto
explícito entre esa instrucción operativa y el procedimiento escrito en la memoria.
Debe acordarse y documentarse el procedimiento definitivo; no se puede declarar
equivalencia de las métricas actuales sin comprobarlo en el equipo experimental.
El monitor empieza después de cargar el modelo y mide cada llamada: tampoco
registra una línea base anterior a la carga ni un máximo de toda la ejecución.
Hay que precisar qué abarca cada «prueba» al informar memoria.

### 2. Una solución ausente impide evaluar un enunciado presente

**Prioridad alta.** `main.tex:619` exige evaluar primero el enunciado y registrar
la solución ausente por separado. `Problem.solucion` exige texto no vacío
(`common/contracts.py:24`). Una respuesta con enunciado válido y solución vacía
falla en `common/generation.py:95`; `judge/evaluate.py:129` la omite por completo.
Así no se obtienen las cinco dimensiones de ese enunciado ni se contabiliza su
solución ausente entre los resolubles. Debe distinguirse un fallo de formato de
la ausencia de solución cuando el enunciado sí se puede recuperar de forma fiable.

### 3. La separación permite versiones del mismo candidato en ambos splits

**Prioridad alta.** `common/data.py:18` considera identidad al par `(id, version)`.
Dos versiones del mismo problema con redacción distinta pueden quedar una en
train y otra en validación sin activar el detector literal. `prepare` también
admite ambas versiones como ejemplos distintos. Esto compromete la independencia
de la validación prevista en `main.tex:812–816`. Debe exportarse una sola versión
aceptada por candidato y rechazarse el solapamiento de sus IDs entre particiones.
La revisión de paráfrasis sigue siendo necesaria además de ese control.

### 4. Ejecutar generación no exige el benchmark completo ni su identidad curricular

**Prioridad media.** Los tamaños 59/295 se verifican en `prepare`, pero
`common/generation.py:17` acepta directamente cualquier archivo no vacío con el
contrato Request. Es posible ejecutar referencia con una sola solicitud sin aviso
de que no corresponde al benchmark final. Tampoco existe un registro local de los
59 códigos seleccionados, su distribución por curso/eje o su versión para
comprobarlos: contar 59 códigos arbitrarios no demuestra correspondencia con la
tabla de selección de la memoria. Conviene validar el benchmark final en cada
ejecución y distinguir explícitamente las solicitudes de desarrollo.

### 5. Los reportes presentan el ajuste a la solicitud como una sexta dimensión

**Prioridad media.** `judge/report.py:9` reúne todas las claves de `statement`,
incluyendo `request_compliance`, bajo `dimensions`. `main.tex:926` exige informarlo
por separado. No contamina `joint_pass`, pero la salida del resumen mezcla ambos
conceptos. Debe tener su propio campo fuera de las cinco dimensiones.

### 6. Desarrollo, calibración y revisión docente quedan fuera del flujo implementado

**Pendiente metodológico.** `main.tex:759`, `:808` y `:918` requieren desarrollo
separado, calibración del juez y comprobaciones previas de la skill. No hay un
registro de desarrollo en la configuración ni una validación automática de su
separación respecto del benchmark. La skill incluida sigue siendo provisional
y su cargador no bloquea usarla en el benchmark.

`export-human` exporta todas las generaciones en orden; no selecciona las muestras
previstas: 118 problemas del dataset (dos por OA, uno few-shot y otro restante;
`main.tex:743`) ni la muestra final estratificada por modelo/estrategia de al menos
30 problemas por modelo (`:918`). Tampoco calcula acuerdo por dimensión y conjunto,
acuerdo de solución entre resolubles por ambos evaluadores, ni resoluciones de
discrepancias. El README encarga la selección al investigador, por lo que es trabajo
manual pendiente y no evidencia de que esas revisiones ya se hayan realizado.

### 7. Hay decisiones adicionales de implementación que deben documentarse

- El juez realiza **tres llamadas** para casos resolubles: puntuar enunciado,
  resolver independientemente y contrastar solución. `main.tex:916` describe dos
  etapas. La verificación independiente es coherente con el objetivo de evitar
  sesgo, pero esta llamada adicional no está explicitada allí y afecta costo/tiempo.
- Se fuerza JSON mediante decodificación estructurada (`common/inference.py:97`).
  La memoria exige campos separados, pero no establece ese mecanismo. Conviene
  registrarlo como condición común porque restringe la generación.
- `target_modules=["all-linear"]` es una aproximación general a atención y MLP.
  El código verifica que solo haya parámetros LoRA entrenables y registra sus
  nombres, pero no comprueba que cubran exactamente todos los módulos previstos
  y excluyan cualquier otro módulo en cada arquitectura. No está validado todavía
  con los seis modelos.
- La rúbrica draft añade matices a la tabla de `main.tex:653–661`: condiciones
  curriculares en alineación y una indicación sobre magnitud de números en
  adecuación pedagógica. No están aprobados. También conviene trasladar
  explícitamente la regla de `main.tex:619` sobre errores lingüísticos menores.
  No se modificaron estos descriptores en la auditoría.

### 8. La trazabilidad aún no acredita completamente la ejecución reproducible

**Prioridad media.** `common/files.py:55` registra versiones principales y hashes
del código, pero no conserva el contenido del código ni la plantilla efectiva del
tokenizador o los IDs de tokens enviados. `model.revision` acepta cualquier texto,
incluso el placeholder de las configuraciones de ejemplo: no acredita que los
archivos de una carpeta local correspondan realmente a ese commit. La procedencia
LoRA solo compara la cadena de revisión al cargar el adaptador, no el identificador
del modelo base (`common/generation.py:40`).

La memoria pide versiones de TRL y contenedor y registrar limitaciones de
determinismo (`main.tex:892`). El proyecto usa Trainer propio con Transformers,
no TRL; debe documentarse que no se utiliza. No se guarda identidad del contenedor
ni se persisten expresamente las advertencias de determinismo. El hardware de
entrenamiento registra GPU/CUDA/Slurm; el de inferencia carece de un registro
equivalente de GPU/controlador y condiciones de carga.

## Valores presentes en el proyecto que la memoria no fija numéricamente

No son discrepancias numéricas demostradas: falta fijarlos mediante desarrollo
y documentarlos antes del benchmark. La memoria pide mantenerlos constantes.

| Parámetro | Configuración actual |
| --- | --- |
| Temperatura del generador | 0.6 |
| top_p del generador | 0.95 |
| Reserva máxima de salida del generador | 1024 tokens |
| Repeticiones | 1; insuficiente por sí sola para la repetición de mediciones prevista |
| Semilla de generación | 42 + índice de repetición |
| gpu_memory_utilization | 0.85 |
| Contexto / salida del juez | 8192 / 2048 tokens |
| Temperatura / top_p del juez | 0 / 1; las llamadas fuerzan temperatura 0 |
| Thinking | `enable_thinking: false` en el ejemplo SmolLM3; falta fijar las plantillas por modelo |
| Precisión en inferencia | BF16 |
| Concurrencia y prefix caching | Una secuencia, un dispositivo, prefix caching desactivado |
| Muestreo de RAM/swap/RSS | Cada 100 ms; RSS puede contar memoria compartida varias veces |

`seed` y `learning_rate` son configurables sin restringirse en entrenamiento a 42
y las tres tasas. Los defaults coinciden y la selección exige las tres tasas,
pero se pueden lanzar entrenamientos ajenos al protocolo. Lo mismo ocurre con
el modelo y los módulos adaptados: el catálogo es informativo, no una restricción
del ejecutor. Se necesita congelar una configuración final común verificable.

## Evidencia y límites de la comprobación

- Lectura del `main.tex` y del código/configuraciones/documentación del proyecto.
- `uv run --locked --no-sync pytest -q`: **20 pruebas aprobadas**.
- `uv run --locked --no-sync ruff check .`: aprobado.
- `uv run --locked --no-sync ruff format --check .`: 30 archivos conformes.
- Se usó `uv --cache-dir /tmp/test-slms-uv-cache` para mantener la caché dentro
  del espacio de escritura permitido. No se sincronizaron dependencias.
- Comprobaciones adicionales con objetos sintéticos: se aceptan versiones
  distintas del mismo ID entre train/validation; `build_requests` acepta un
  benchmark de una solicitud; `Problem` rechaza solución vacía; el resumen
  incluye `request_compliance` dentro de `dimensions`.

No hay datos reales ni configuraciones definitivas en `data/` y `configs/`:
solo documentación, ejemplos y rúbrica draft. Por tanto, esta auditoría comprueba
la implementación y sus valores previstos, no la calidad del dataset, los códigos
OA efectivos, las revisiones de pesos, resultados experimentales o compatibilidad
de tokenizadores/GPU. No se ejecutó entrenamiento, inferencia ni evaluación con
modelos, no se descargaron pesos y no se consultó la planilla externa de OA.
