# Mediciones con ejecución local

El mismo comando carga los pesos con `vllm.LLM` y genera mediante `LLM.generate`.
No se inicia un servidor ni se usan solicitudes HTTP. Se mantiene una única GPU,
una solicitud por llamada y prefix caching desactivado. El modelo se carga una vez
por ejecución. El tiempo de carga no se incluye en la latencia por problema.

Se conservan estas métricas por generación:

| Campo | Definición |
| --- | --- |
| `usage.prompt_tokens` | Cantidad de tokens de la entrada con la plantilla aplicada |
| `usage.completion_tokens` | Cantidad de tokens producidos por vLLM |
| `latency_s` | Tiempo de pared de la llamada local a `LLM.generate` |
| `output_tokens_per_wall_second` | Tokens de salida divididos por `latency_s`, incluyendo prefill |
| `ttft_s` | Primer token menos llegada, solo si vLLM proporciona esos tiempos |
| `decode_s` | Último token menos primer token, cuando existen tiempos válidos |
| `decode_tokens_per_second` | `(output_tokens - 1) / decode_s`, con más de un token |

La disponibilidad de `RequestOutput.metrics` depende de la versión y del motor de
vLLM. Si no hay tiempos por token, los campos correspondientes se guardan como
`null` y `timing_source` como `unavailable`. No se estima TTFT a partir de la
latencia total. La tasa que incluye prefill no equivale a la velocidad de decode.

La memoria se muestrea cada 100 ms, antes y durante cada llamada: RAM usada y swap
del host, más la suma de RSS del proceso local y sus hijos, incluidos workers de
vLLM. Se guardan valor inicial, máximo observado y errores de lectura. La suma de
RSS puede contar páginas compartidas más de una vez. No representa memoria GPU
ni garantiza capturar picos más breves que el intervalo de muestreo.

En GB10, interpretar estos datos junto a los registros del sistema, por su memoria
unificada. Mantener configuración, indicaciones, decodificación, repeticiones y
condiciones de carga comunes. Documentar calentamiento, efectos de compilación,
versiones, hardware y límites de memoria antes de comparar resultados.

## Correspondencia con el informe

La metodología actual del informe menciona `vllm bench serve`. Por la decisión de
ejecutar directamente en el computador experimental, este repositorio usa la API
Python offline y no incluye ese flujo de servidor. Si se adopta este procedimiento
para las mediciones finales, actualizar esa subsección del informe y comprobar qué
métricas temporales expone la versión fijada. Mientras no se disponga de ellas,
TTFT y velocidad de decode permanecen sin medir.
