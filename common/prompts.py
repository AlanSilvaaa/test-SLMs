import json

SYSTEM = (
    "Genera problemas matemáticos verbales en español. Respeta la solicitud, "
    "el curso y el OA. El enunciado debe tener una respuesta única y verificable. "
    'Responde solo JSON con dos campos: "enunciado" y "solucion". '
    "En solucion explica el procedimiento e incluye la respuesta final con sus unidades."
)


def user_prompt(row):
    return json.dumps(
        {
            key: row[key]
            for key in (
                "curso",
                "codigo_oa",
                "texto_oa",
                "habilidad_evaluada",
                "condiciones_curriculares",
                "contexto",
                "solicitud",
            )
        },
        ensure_ascii=False,
    )


def conversation(row, examples=(), instructions=""):
    messages = [
        {"role": "system", "content": SYSTEM + ("\n\n" + instructions if instructions else "")}
    ]
    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": user_prompt(example)},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"enunciado": example["enunciado"], "solucion": example["solucion"]},
                        ensure_ascii=False,
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": user_prompt(row)})
    return messages
