"""Command line entry point. Importing the CLI never loads a model."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from common.config import load_config
from common.environment import load_repo_env
from common.models import DEFAULT_MODELS_DIR

app = typer.Typer(
    no_args_is_help=True, help="Experimentos SLM para generación de problemas matemáticos"
)


def show_run(run):
    excel = Path(run) / "resultados.xlsx"
    typer.echo(f"Excel: {excel}" if excel.is_file() else run)


@app.callback(invoke_without_command=True)
def quick(
    ctx: typer.Context,
    count: Annotated[
        int,
        typer.Option(
            "-n", "--num-problems", min=1, help="Cantidad de solicitudes de la prueba base"
        ),
    ] = 5,
    model: Annotated[
        str, typer.Option("--model", "-model", "-m", help="Alias o modelID; por defecto SmolLM3")
    ] = "SmolLM3",
    models: Annotated[
        str | None,
        typer.Option(
            "--models", "-models", help="Usa all para ejecutar los seis modelos en secuencia"
        ),
    ] = None,
    models_dir: Annotated[
        Path, typer.Option("--models-dir", help="Caché local de pesos por modelID")
    ] = DEFAULT_MODELS_DIR,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Commit exacto opcional")
    ] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Usar solo pesos completos ya locales")
    ] = False,
    requests: Annotated[
        Path,
        typer.Option(
            "--requests", exists=True, dir_okay=False, help="Solicitudes de desarrollo JSONL"
        ),
    ] = Path(__file__).resolve().parent / "examples/smoke/requests.jsonl",
    output: Annotated[
        Path, typer.Option("--output", help="Carpeta de resultados de la prueba")
    ] = Path("output"),
    port: Annotated[
        int, typer.Option("--port", min=1024, max=65535, help="Puerto local de vLLM")
    ] = 8000,
):
    """Prueba rápida base: uv run main.py -n 5 --model SmolLM3."""
    load_repo_env()
    if ctx.invoked_subcommand is not None:
        names = (
            "count",
            "model",
            "models",
            "models_dir",
            "revision",
            "offline",
            "requests",
            "output",
            "port",
        )
        if any(ctx.get_parameter_source(name).name == "COMMANDLINE" for name in names):
            raise typer.BadParameter(
                "Los flags de prueba rápida se usan sin subcomando; "
                "para el experimento completo usa run --config"
            )
        return
    from common.quick import run_quick

    selected = [model]
    if models is not None:
        if models.casefold() != "all":
            raise typer.BadParameter("--models solo admite all; para uno usa --model ALIAS")
        if ctx.get_parameter_source("model").name == "COMMANDLINE":
            raise typer.BadParameter("Usa --models all o --model ALIAS por separado")
        if revision is not None:
            raise typer.BadParameter("--revision corresponde a un solo modelo; usa --model ALIAS")
        from common.files import read_json

        selected = read_json(Path(__file__).resolve().parent / "configs/models.json")["generators"]
        typer.echo(f"Prueba base: {len(selected)} modelos, {count} solicitudes por modelo.")
    if not offline:
        typer.echo("Se reutilizarán los pesos locales; si faltan, se descargarán antes de generar.")
    for index, selected_model in enumerate(selected, 1):
        typer.echo(
            f"[{index}/{len(selected)}] Prueba base: {selected_model}, {count} solicitudes; "
            f"pesos en {models_dir}."
        )
        show_run(
            run_quick(count, selected_model, models_dir, revision, offline, requests, output, port)
        )


class Module(StrEnum):
    base = "base_model"
    few_shot = "few-shot"
    skill = "skill"
    lora = "lora"


class Action(StrEnum):
    infer = "infer"
    train = "train"


@app.command()
def run(
    module: Annotated[Module, typer.Option("--module", help="Estrategia individual")],
    action: Annotated[Action, typer.Option("--action", help="Operación individual")],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    shots: Annotated[int, typer.Option(min=1, max=4)] = 4,
):
    cfg = load_config(config)
    if action == Action.train:
        if module != Module.lora:
            raise typer.BadParameter("Solo LoRA requiere entrenamiento")
        from lora.train import train

        result = train(cfg)
    else:
        from common.generation import infer

        result = infer(cfg, module.value, shots)
    show_run(result)


@app.command("export-excel")
def export_excel(
    run_dir: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
):
    """Crea un Excel de una ejecución existente sin cargar modelos."""
    from common.excel import export_run

    typer.echo(export_run(run_dir, output))


@app.command("prepare-pilot")
def prepare_pilot_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    oa_source: Annotated[Path, typer.Option("--oa-source", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("data/pilot"),
):
    """Prepara solicitudes piloto desde los Excel, sin generar ni aprobar datos."""
    from common.pilot_data import prepare_pilot

    typer.echo(prepare_pilot(source, oa_source, output))


@app.command("pilot")
def pilot_command(
    data: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("data/pilot"),
    output: Annotated[Path, typer.Option()] = Path("output"),
    models_dir: Annotated[Path, typer.Option("--models-dir")] = DEFAULT_MODELS_DIR,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    port: Annotated[int, typer.Option(min=1024, max=65535)] = 8000,
    background: Annotated[
        bool, typer.Option("--background", help="Continuar aunque se cierre SSH")
    ] = False,
):
    """Genera las 295 solicitudes piloto con los seis modelos base, una pasada."""
    from common.pilot import launch_pilot, run_pilot

    if background:
        pid, log = launch_pilot(data, output, models_dir, offline, port)
        typer.echo(f"Proceso lanzado en segundo plano. PID: {pid}")
        typer.echo(f"Registro: {log}")
        typer.echo(f'Ver avance: tail -f "{log}"')
    else:
        run_pilot(data, output, models_dir, offline, port, report=typer.echo)


@app.command()
def prepare(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    benchmark: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("output/splits"),
):
    """Genera particiones 34/6 y cuatro demostraciones por OA, con semilla 42."""
    from common.data import prepare as prepare_data

    typer.echo(prepare_data(dataset, benchmark, output))


@app.command()
def judge(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    rubric: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
):
    """Evalúa enunciados y soluciones con un modelo juez cargado localmente."""
    from judge.evaluate import evaluate

    typer.echo(evaluate(load_config(config), source, rubric))


@app.command("select-lora")
def select_lora(
    runs: Annotated[list[Path], typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("selected-adapter.json"),
):
    """Selecciona por pérdida, época y learning rate entre tres entrenamientos."""
    from lora.train import select

    select(runs, output)
    typer.echo(output)


@app.command()
def summarize(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path("summary.json"),
):
    from judge.report import summarize as report

    report(source, output)
    typer.echo(output)


@app.command("export-human")
def export_human(
    source: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    sample: Annotated[bool, typer.Option("--sample/--all")] = True,
):
    """Exporta formularios separados y un mapa privado para revisión docente."""
    from judge.report import export_human as export

    export(source, output, sample=sample)
    typer.echo(output)


@app.command("import-benchmark")
def import_benchmark(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    """Convierte solicitudes de MathGenBench-CL sin copiar sus resultados."""
    from common.import_benchmark import convert

    convert(source, output)
    typer.echo(output)


@app.command("sample-dataset")
def sample_dataset(
    train: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    validation: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    demonstrations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    """Selecciona los 118 problemas para la revisión docente del dataset."""
    from judge.sampling import sample_dataset as sample

    sample(train, validation, demonstrations, output)
    typer.echo(output)


@app.command("agreement")
def agreement(
    judge: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
    human: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mapping: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    """Compara revisiones independientes sin sobrescribir los originales."""
    from judge.agreement import compare

    compare(judge, human, mapping, output)
    typer.echo(output)


def main():
    try:
        app()
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
