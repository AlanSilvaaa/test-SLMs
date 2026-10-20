# Datos locales

Archivos UTF-8 JSONL, un objeto por línea. Se admite también un array JSON.
Los contratos completos están en `common/contracts.py`. No se distribuyen
problemas inventados como si fueran ejemplos aceptados.

Cada solicitud requiere `id`, `version` (entero desde 1), `curso`, `codigo_oa`,
`texto_oa`, `habilidad_evaluada`, `condiciones_curriculares`, `contexto` y
`solicitud` (instrucción completa). Todos los textos deben ser no vacíos.

Cada ejemplo añade `enunciado`, `solucion` (procedimiento y respuesta final),
`procedencia` y `estado: "aceptado"`. Se exportan del banco únicamente las
versiones aceptadas tras la revisión humana. No incluir candidatos pendientes
ni rechazados. Conservar sus evaluaciones en el banco de origen.

`prepare` genera archivos inmutables en una carpeta de ejecución. Copiar sus
rutas en la configuración o copiar `train.jsonl`, `validation.jsonl` y
`demonstrations.json` aquí. El archivo de demostraciones mapea cada código OA
a cuatro objetos `{"id": "...", "version": 1}` ordenados como A, B, C, D.
Puede ajustarse manualmente antes de fijarlo, conservando cuatro ejemplos del
train por OA y documentando la selección pedagógica.

La detección automática bloquea duplicados normalizados, IDs compartidos y
solicitudes idénticas reservadas. La detección de reformulaciones superficiales
y la aceptación curricular requieren la revisión prevista en el informe.
