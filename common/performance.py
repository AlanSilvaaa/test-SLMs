"""Keep aggregate throughput separate from per-problem decoding speed."""

from statistics import mean, median, pstdev

from common.files import load_rows, read_json, write_json


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "std": pstdev(values) if values else None,
    }


def summarize_performance(source, output):
    rows = load_rows(source)
    metrics = ("ttft_s", "latency_s", "decode_s", "tpot_s", "decode_tokens_per_second")
    repetitions = {}
    for repetition in sorted({r["repetition"] for r in rows}):
        group = [r for r in rows if r["repetition"] == repetition]
        native = read_json(group[0]["bench_result"])
        repetitions[str(repetition)] = {
            "aggregate_output_tokens_per_second": native["output_throughput"],
            "benchmark_duration_s": native["duration"],
            "completed": native["completed"],
            "failed": native["failed"],
            "per_problem": {
                key: distribution([r.get(key) for r in group if not r["error"]]) for key in metrics
            },
        }
    write_json(
        output,
        {
            "repetitions": repetitions,
            "aggregate_output_tokens_per_second": distribution(
                [r["aggregate_output_tokens_per_second"] for r in repetitions.values()]
            ),
            "per_problem": {
                key: distribution([r.get(key) for r in rows if not r["error"]]) for key in metrics
            },
            "output_limit_reached": sum(r["output_limit_reached"] for r in rows),
            "parse_errors": sum(bool(r.get("parse_error")) for r in rows),
        },
    )
