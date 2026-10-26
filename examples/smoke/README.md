# Prueba técnica: SmolLM3, condición base, cinco problemas

Las cinco solicitudes de `requests.jsonl` son sintéticas y públicas. El código
`DEV-SUMA-20` y su objetivo son de prueba; no representan un OA oficial, un dataset
aceptado ni solicitudes del benchmark final. Permiten comprobar carga, prompts,
salida JSON, mediciones y guardado sin preparar LoRA, few-shot, skill o judge.

La forma más sencilla, en el computador experimental y desde la raíz del clon:

Si necesitas un token, copiar `.env.example` a `.env` en la raíz y completar
`HF_TOKEN=hf_TU_TOKEN`. Se carga automáticamente y no hace falta ejecutar un login.

```bash
uv run --extra infer main.py -n 5 --model SmolLM3
```

Para probar otro modelo, reemplazar `SmolLM3` por `llama-1b`, `granite-2b`,
`phi4-mini`, `qwen3-4b` o `granite-8b`. La [tabla de alias](../../README.md#prueba-rápida-con-flags)
indica el modelID exacto de cada uno. Cada comando ejecuta solo el modelo elegido.

Busca el modelo en `models/` y lo descarga allí si falta. La estructura conserva
el modelID y el commit: `models/models--HuggingFaceTB--SmolLM3-3B/snapshots/<commit>/`.
Usar `--offline` para prohibir descargas o `--models-dir` para cambiar la carpeta.
No se necesita copiar ni editar configuración. Se usa una repetición y la condición
base, con las cinco solicitudes de este directorio. Para N mayor que cinco,
proporcionar suficientes solicitudes mediante `--requests`; no se repiten entradas.

La alternativa con archivo de configuración sigue disponible:

```bash
uv sync --locked --extra infer
cp configs/smollm3-smoke.example.json configs/smollm3-smoke.json
```

Editar `model.path` con la carpeta absoluta de los pesos y tokenizador locales,
y `model.revision` con el commit exacto de Hugging Face (40 caracteres). Si los
pesos están en una carpeta `snapshots/<commit>`, el nombre de esa carpeta identifica
la revisión descargada. Los pesos deben estar completos y disponibles localmente.

```bash
uv run --locked --no-sync main.py run --module base_model --action infer \
  --config configs/smollm3-smoke.json
```

Se ejecuta una repetición con cinco solicitudes, solo SmolLM3, sin ejemplos ni
adaptación. Se inician automáticamente serve y bench sobre loopback. vLLM realiza
una solicitud inicial adicional de comprobación; los cinco resultados del conjunto
se guardan en `output/HuggingFaceTB/SmolLM3-3B/<fecha-hora>/resultados.xlsx`, con
una fila por generación y una hoja de resumen. El comando imprime esa ruta.
La subcarpeta `detalles/` conserva `generations.jsonl`, `performance.json`,
resultados nativos `bench-0001/result.json` y logs.
Un resultado puede registrar un error técnico o de formato: revisar `error`,
`parse_error` y `output_limit_reached`, además de `problem` y `raw_response`.

No se requiere el registro OA ni el benchmark para esta prueba de desarrollo.
Si existe `data/benchmark.jsonl`, se comprueba separación con estas solicitudes;
si no existe, se registra `benchmark_separation_checked: false`. El modo
`benchmark` mantiene sus requisitos completos. Esta prueba no certifica
compatibilidad GPU: debe ejecutarse en el equipo experimental.
