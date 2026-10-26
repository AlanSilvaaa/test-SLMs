# Instrucciones comunes

Este repositorio implementa el experimento de la memoria sobre generación de
problemas matemáticos para educación básica chilena con SLM offline.

- Usar `uv add`, `uv sync` y `uv run`. Mantener `uv.lock` actualizado.
- No ejecutar entrenamiento, inferencia, evaluación ni descargar pesos en este
  computador. Esas tareas se realizan en el hardware experimental.
- Se permiten lint y pruebas unitarias sin modelos ni GPU.
- Mantener cada estrategia en su carpeta. Compartir contratos, prompts, lectura,
  inferencia local con vLLM, métricas y trazabilidad en `common/`. El judge es independiente.
- Ejecutar generación y judge con los comandos oficiales `vllm serve` y
  `vllm bench serve` en el computador experimental donde se clona el repositorio.
  Limitar el servidor a `127.0.0.1`. No usar la API Python de inferencia, clientes
  HTTP propios ni servicios externos. La memoria prevalece ante contradicciones.
- Importar dependencias de ML solo dentro de las funciones que las necesitan.
- No aprobar ni cambiar silenciosamente la rúbrica. Versionar sus descriptores,
  conservar la versión usada y distinguir draft de approved.
- Evaluar el enunciado sin solución ni identidad del generador. Comprobar después
  la solución solo si el enunciado es resoluble. No agregar corrección al indicador
  conjunto de las cinco dimensiones. Registrar ajuste a la solicitud aparte.
- Preservar dataset, validación, desarrollo y benchmark separados. No usar
  generaciones del benchmark como entrenamiento ni demostraciones.
- Mantener los splits 34/6 por OA, semilla 42, y demostraciones únicamente de train.
  Los detectores de duplicados literales no sustituyen la revisión pedagógica ni
  la búsqueda de paráfrasis antes de aceptar candidatos.
- Nunca truncar silenciosamente, sobrescribir ejecuciones ni asumir que un error
  de parseo es una valoración pedagógica negativa.
- Documentar cambios metodológicos frente al informe. No inventar mediciones ni
  afirmar compatibilidad GPU sin probarla en el equipo correspondiente.
- No guardar tokens, cachés de modelos, datasets privados ni resultados en Git.
- La fuente de la verdad será lo que diga el `main.tex` de la memoria de titulo, ubicado en `/home/alan/Documents/memoria-de-titulo/main.tex`
