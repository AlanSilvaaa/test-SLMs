# Medición con vllm bench serve

El procedimiento vigente sigue `main.tex`: un proceso `vllm serve` y el benchmark
oficial `vllm bench serve`, ambos en el mismo computador experimental. El servidor
solo escucha en `127.0.0.1`; el proyecto no implementa clientes HTTP ni usa
`LLM.generate`. La comunicación loopback es parte del benchmark oficial.

La integración está fijada a vLLM 0.15.1. El dataset `custom` contiene las entradas
reales con la plantilla de conversación ya aplicada. Se fija
`--skip-chat-template --disable-shuffle --no-oversample --max-concurrency 1`,
además de los parámetros de decodificación, el número exacto de solicitudes y
`--save-result --save-detailed`. No se usa `--ignore-eos`, entradas aleatorias ni
recorte de prompts. `add_special_tokens=false` evita duplicar los tokens especiales
que ya contiene la plantilla. Se guardan texto y tokens esperados antes de iniciar
el servidor; se verifica su equivalencia con el tokenizador local.

## Métricas

| Campo | Definición |
| --- | --- |
| `usage.prompt_tokens` | Longitud de la entrada exacta guardada, con plantilla y sin volver a añadir tokens especiales |
| `bench_input_tokens` | Conteo nativo del dataset custom; su tokenización puede añadir especiales y diferir del conteo anterior |
| `usage.completion_tokens` | `output_lens` del resultado nativo; vLLM prioriza usage del servidor y puede retokenizar el texto si no recibió usage |
| `ttft_s` | `ttfts` del benchmark, desde envío hasta primer fragmento de salida |
| `decode_s` | Suma de `itls` nativos: tiempo entre primer y último fragmento de salida |
| `latency_s` | `ttft_s + decode_s`, equivalente al tiempo de la solicitud hasta el último fragmento medido por este backend |
| `tpot_s` | `decode_s / (completion_tokens - 1)`, con más de un token y tiempo positivo |
| `decode_tokens_per_second` | `(completion_tokens - 1) / decode_s` |
| `aggregate_output_tokens_per_second` | `output_throughput` nativo: tokens de salida completados / duración del benchmark |

Los fragmentos de streaming pueden agrupar tokens. Son mediciones observadas por
el benchmark, no marcas internas del motor para cada token. Las respuestas fallidas
mantienen su posición y error; sus tiempos derivados quedan ausentes. Se rechazan
resultados sin arrays detallados completos para evitar asociar métricas a otras
solicitudes. El JSON nativo se conserva sin modificar, incluyendo sus métricas
agregadas de TTFT, TPOT, latencia y percentiles.

El benchmark no guarda `finish_reason`. Se conserva como null y se registra
`output_limit_reached` si los tokens alcanzan el presupuesto de salida. Este aviso
no se oculta ni se interpreta automáticamente como una valoración pedagógica.
La salida original se conserva y, si es un objeto JSON válido con enunciado,
puede evaluarse su calidad. Un fallo de parseo sigue siendo un fallo técnico.

`performance.json` informa media, mediana y desviación estándar poblacional de las
métricas por problema, por repetición y agrupadas; la tasa agregada se resume entre
repeticiones y no se confunde con la velocidad de decode. El benchmark final exige
al menos dos repeticiones; el ejemplo usa tres, con semillas 42, 43 y 44.

## Calentamiento, memoria y ejecución

vLLM carga una sola vez por comando. Cada invocación de bench realiza su comprobación
inicial con la primera entrada antes de medir el conjunto, según el mecanismo
oficial `--ready-check-timeout-sec`. Esa respuesta inicial no se incluye en el
conjunto evaluado. No se añaden warmups: `--num-warmups 0`. El prefix caching está
desactivado. La carga del servidor no forma parte de las latencias por problema.
Documentar estas condiciones al comparar resultados.

`memory.json` de la ejecución mide antes de cargar el servidor y hasta completar
las repeticiones. Cada carpeta `bench-*` contiene, además, la línea base y el máximo
durante esa invocación, incluida su comprobación inicial. La RAM y el swap del host
y el RSS del proceso y sus descendientes (servidor, workers y bench) se muestrean
cada 100 ms. No se atribuye ese máximo a un problema individual. La suma de RSS
puede duplicar páginas compartidas y no captura necesariamente picos más breves.
En GB10, interpretar estos datos como memoria unificada; `nvidia-smi` se utiliza
para identificar hardware/controlador y no como estimación exclusiva del consumo.

Se conservan versión de sistema y paquetes, identidad declarada del contenedor,
GPU/controlador, RAM, carga inicial y descripción de condiciones de carga. Esto no
certifica que el equipo estuviera aislado: debe prepararlo el investigador.

Las carpetas de resultados no se reutilizan. Se guardan comandos y logs incluso
cuando falla un proceso. El servidor y sus hijos se cierran al finalizar o al
propagarse una interrupción. El benchmark escribe el resultado al terminar una
invocación: una interrupción puede perder sus respuestas en curso, mientras las
repeticiones ya finalizadas permanecen guardadas. No hay finalización declarada
si falta completar el flujo.

## Referencias de implementación

- [CLI de benchmark de vLLM 0.15.1](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/benchmarks/serve.py)
- [Dataset custom y sus opciones](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/benchmarks/datasets.py)
- [Medición del streaming del backend](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/benchmarks/lib/endpoint_request_func.py)

Las pruebas del repositorio simulan comandos y resultados. La ejecución completa,
las plantillas reales y la compatibilidad con la GPU se validan en el computador
experimental, nunca se dan por demostradas mediante estas pruebas.
