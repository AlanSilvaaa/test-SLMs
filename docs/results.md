# Resultados de generación

Al completar una generación, el comando imprime la ruta del Excel. Por ejemplo:

```text
output/
└── HuggingFaceTB/
    └── SmolLM3-3B/
        └── 2026-10-26_16-21-34/
            ├── resultados.xlsx
            └── detalles/
                ├── generations.jsonl
                ├── performance.json
                ├── manifest.json
                ├── serve.log
                └── …
```

Se conserva el modelID completo, incluido su propietario. La fecha usa
`America/Santiago`; si dos ejecuciones comienzan en el mismo segundo se añade
`-02`, `-03`, etc. `--output` cambia la carpeta base. No se sobrescriben ejecuciones.

El Excel contiene:

- **Resultados**: una fila por generación y repetición, con curso, OA, enunciado,
  solución, solicitud, estrategia, tokens, tiempos, errores y respuesta original.
  Tiene filtros y encabezados inmovilizados. La repetición se muestra desde 1.
- **Resumen**: configuración, revisión del modelo, versiones, conteos técnicos,
  métricas agregadas y memoria. Las unidades están indicadas; una celda vacía
  representa un dato ausente. La velocidad de decode y la tasa agregada son
  mediciones distintas.
- **Textos largos**, si hace falta: conserva por partes el texto que supera el
  límite conservador por celda. La celda original indica la referencia. Los
  caracteres de control incompatibles con Excel se representan como escapes
  Unicode; los JSON originales conservan su contenido exacto.

Los estados son técnicos. No certifican calidad pedagógica ni sustituyen al judge.
Los archivos de trazabilidad y resultados originales permanecen en `detalles/`.
El cambio afecta la presentación y las rutas, sin modificar generación ni medición.
El judge y el entrenamiento conservan sus propios artefactos, sin este Excel de
generación automático.

## Convertir una ejecución anterior

Después de actualizar el repositorio y sincronizar sus dependencias en el equipo
experimental (`uv sync --locked --extra infer`), no hace falta repetir inferencia:

```bash
uv run --no-sync main.py export-excel \
  --run output/smoke/20261026T192134Z-base_model-fe6a74e5
```

Crea `resultados.xlsx` dentro de esa carpeta antigua, sin mover sus archivos.
También admite el nuevo formato con `detalles/`. Para otro destino:

```bash
uv run --no-sync main.py export-excel --run output/MODEL/FECHA \
  --output output/mi-revision.xlsx
```

El destino debe ser nuevo. Una ejecución interrumpida puede exportarse si conserva
su `manifest.json`; se incluyen solo las generaciones guardadas y se advierte su
estado incompleto. La exportación automática ocurre únicamente al completar el
flujo de generación.
