"""Command line entry point. Importing the CLI never loads a model."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from common.config import load_config

app = typer.Typer(
    no_args_is_help=True, help="Experimentos SLM para generación de problemas matemáticos"
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
    typer.echo(result)


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
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
):
    """Exporta formularios separados y un mapa privado para revisión docente."""
    from judge.report import export_human as export

    export(source, output)
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


def main():
    try:
        app()
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
