# Entorno CUDA de la GX10 (ARM64/GB10)

## Diagnóstico observado el 26 de octubre de 2026

El equipo experimental informó `aarch64`, GPU NVIDIA GB10 y controlador
580.173.02. `nvidia-smi` informó CUDA 13.0 (capacidad del controlador; no demuestra
que esté instalado un runtime determinado). La instalación del entorno del
proyecto mostró **PyTorch 2.9.1+cpu**, `torch.version.cuda=None` y
`torch.cuda.is_available()=False`. Al importar la extensión de vLLM apareció
`ImportError: libcudart.so.12`.

La distribución CPU de PyTorch no proporciona el runtime GPU que requiere esa
extensión. No basta con instalar el paquete genérico desde PyPI en este equipo.
No se deben fabricar enlaces entre `libcudart.so.12` y `libcudart.so.13`.

## Dependencias corregidas en el proyecto

Para Linux ARM64, `pyproject.toml` y `uv.lock` seleccionan:

- PyTorch **2.9.1+cu130** desde el índice oficial CUDA 13 de PyTorch.
- torchvision **0.24.1** y torchaudio **2.9.1** desde ese mismo índice.
- El wheel oficial **vLLM 0.15.1+cu130** para ARM64, publicado en GitHub Releases.
- Los paquetes de runtime CUDA indicados por esos builds, con hashes en el lockfile.

La selección por plataforma conserva la distribución previa para x86_64. No se
instalan estos extras en el computador de desarrollo. El wheel ARM64 de vLLM
requiere glibc >= 2.35. La resolución del lockfile y el plan de instalación se
comprobaron para `aarch64-manylinux_2_35`; esto no prueba ejecución de kernels ni
compatibilidad de todos los modelos en la GX10.

El cambio conserva la versión base 0.15.1 y los comandos `serve`/`bench serve`.
La trazabilidad guarda la versión completa `0.15.1+cu130`. No se cambian los
parámetros de entrenamiento, decodificación, prompts ni métricas experimentales.

## Reparación en el computador experimental

Con esta versión del repositorio actualizada en la GX10:

```bash
uv sync --locked --extra infer
```

Este comando debe reemplazar PyTorch CPU y vLLM por las distribuciones seleccionadas
para ARM64/CUDA 13. No borra `models/` ni descarga otra vez el modelo por ese cambio.

Comprobar el resultado, sin realizar generación:

```bash
uv run --no-sync python -c 'import torch; from importlib.metadata import version; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.is_available()); print("vLLM:", version("vllm"))'
uv run --no-sync vllm --version
```

Se espera PyTorch `2.9.1+cu130`, CUDA `13.0`, GPU `True` y vLLM `0.15.1+cu130`.
El segundo comando comprueba que la CLI importe correctamente; no carga un modelo.
Si estas comprobaciones pasan, ejecutar la prueba habitual:

```bash
uv run --no-sync main.py -n 5 --model SmolLM3
```

El programa también verifica que PyTorch tenga CUDA y pueda acceder a la GPU,
antes de preparar descargas y al iniciar el backend. Si aparece otro error de
bibliotecas o kernels, conservar la salida y `serve.log`: aún debe validarse la
ejecución completa en ese equipo, sin asumir que resolver el import garantiza
compatibilidad de la generación.

## Fuentes oficiales verificadas

- [Artefactos de vLLM 0.15.1, incluido ARM64 CUDA 13](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.15.1)
- [Índice CUDA 13 de PyTorch](https://download.pytorch.org/whl/cu130/torch/)
- [Requisitos de vLLM 0.15.1](https://github.com/vllm-project/vllm/blob/v0.15.1/requirements/cuda.txt)
- [Instalación de vLLM y compatibilidad binaria](https://docs.vllm.ai/en/v0.15.1/getting_started/installation/gpu/)
