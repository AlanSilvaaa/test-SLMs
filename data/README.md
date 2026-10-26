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

## Registro curricular y desarrollo

En la ejecución final, `oa_registry` apunta a un JSON local preparado desde el
registro de selección de la memoria, con esta estructura:

```json
{
  "version": "VERSION_REVISADA",
  "status": "approved",
  "objectives": [
    {"codigo_oa": "CODIGO_REAL", "curso": "1", "eje": "numeros", "texto_oa": "TEXTO_OFICIAL_COMPLETO"}
  ]
}
```

El ejemplo muestra solo la estructura; no constituye un registro válido. Se
requieren los 59 OA reales, sin duplicados. `curso` utiliza las cadenas `"1"` a
`"6"`; `eje` utiliza `numeros`, `medicion`, `patrones` o `datos`. Se verifica la
distribución por curso/eje de la tabla de la memoria y que cada solicitud tenga
el curso y texto exactos del OA registrado. Normalizar esos campos al preparar los
datos, sin reemplazar ni resumir el texto oficial. No se incluyen OA inventados.

`development` es otro archivo Request JSONL, con solicitudes diferentes de las
reservadas para benchmark. El generador usa este archivo con
`phase: "development"`. Con `phase: "benchmark"` exige cinco solicitudes por cada
OA registrado, 295 en total, y comprueba la separación con desarrollo.

El dataset aceptado debe contener **una sola versión por ID**. Una nueva versión
conserva el historial en el banco de candidatos, pero no se incorpora junto a la
anterior al dataset exportado ni se distribuye entre train y validación.
La integridad curricular y las paráfrasis siguen requiriendo revisión humana.

Las respuestas de generación tienen un contrato distinto de los ejemplos
aceptados: un enunciado válido puede conservar una solución ausente o vacía para
evaluarla como ausencia separada. Los ejemplos de entrenamiento y demostraciones
siguen exigiendo ambos campos no vacíos y aceptación previa.
