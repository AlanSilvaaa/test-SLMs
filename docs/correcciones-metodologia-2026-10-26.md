# Correcciones posteriores a la auditoría

La auditoría original se conserva como registro histórico. Esta revisión del
26 de octubre de 2026 aplica la decisión explícita del usuario de seguir
`main.tex` y utilizar `vllm bench serve`, ejecutado en el computador experimental.
No modifica la memoria ni atribuye resultados experimentales a pruebas unitarias.

| Hallazgo original | Corrección |
| --- | --- |
| Medición diferente del informe | Sustituida la API Python de inferencia por comandos oficiales `vllm serve` + `vllm bench serve` locales. Resultados detallados, tasa agregada separada, TTFT/TPOT/decode/latencia y resúmenes con dispersión. Mínimo dos repeticiones finales; ejemplo con tres. |
| Contradicción en AGENTS.md | Actualizada la instrucción de ejecución conforme a la solicitud explícita del usuario. |
| Solución ausente bloqueaba enunciado | Contrato de generación separado del de ejemplos aceptados. Conserva enunciado con solución ausente/vacía; evaluación separada y corrección 0 si es resoluble y no hay solución. |
| Versiones del mismo candidato entre splits | Rechazo de IDs repetidos aunque cambien la versión y el enunciado. |
| Benchmark incompleto | Validación en generación final de 295 solicitudes, cinco por OA, registro curricular local revisado y distribución de cursos/ejes. Desarrollo usa una fase explícita separada. |
| Ajuste a la solicitud como sexta dimensión | Campo propio en el resumen; las dimensiones son exactamente las cinco previstas y el conjunto las usa exclusivamente. |
| Desarrollo y revisión docente fuera del flujo | Rutas/fases separadas, comprobaciones de solapamiento, evidencia de revisión de skill y calibración de juez. Muestreo de 118 problemas del dataset y muestra final de al menos 30 por modelo con cobertura curricular/estrategias presentes. Comando de acuerdo y registro separado de discrepancias. |
| Tres llamadas del juez | Dos etapas: enunciado ciego y comprobación posterior de solución únicamente si es resoluble. |
| Descriptores provisionales | Original preservado; nueva versión `0.2-draft` añade explícitamente la regla lingüística de la memoria. Ninguna rúbrica fue aprobada automáticamente. |
| Trazabilidad incompleta | Plantilla efectiva, prompts/tokens, comandos/logs/resultados nativos, copia de fuentes y fingerprints de archivos locales, hardware/controlador y contenedor declarado. Identidad/revisión del modelo base verificadas al usar LoRA. TRL declarado como no utilizado. |
| Entrenamientos fuera del protocolo | Semilla 42, tres tasas permitidas y política all-linear exigidas. Se mantiene el resto de hiperparámetros coincidentes. |

## Límites que requieren el equipo experimental y revisión humana

- Comprobar GB10/ARM64, CUDA y vLLM 0.15.1, y realizar la prueba previa en Patagón.
- Fijar la longitud común y la decodificación con los seis tokenizadores y todas
  las estrategias. Los valores que la memoria no especifica siguen siendo
  propuestas de configuración, no parámetros atribuidos a ella.
- Verificar los módulos LoRA efectivos por arquitectura y las máscaras reales.
- Completar el registro de OA con los códigos/textos reales; validar cada candidato,
  detectar paráfrasis y documentar revisiones, aceptación y reemplazos en el banco.
- Realizar revisión pedagógica y pruebas de la skill, calibración del juez y
  revisión docente independiente. Los registros de aprobación se completan solo
  cuando exista evidencia; el código no puede sustituir esas actividades.
- Conservar la resolución fundamentada de discrepancias como registro adicional,
  sin editar las anotaciones originales.

El HTTP de loopback es inherente a `bench serve`. No hay servicios externos ni un
cliente HTTP implementado en este proyecto. El backend oficial no conserva motivos
de terminación ni resultados parciales por solicitud: se documentan ese límite y
el indicador explícito de presupuesto de salida alcanzado.
