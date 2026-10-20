from common.prompts import conversation


def messages(row, *, examples, assignments, ordinal, shots, **_):
    keys = assignments.get(row["codigo_oa"], [])
    if len(keys) != 4 or len({(k["id"], k["version"]) for k in keys}) != 4:
        raise ValueError("Cada OA debe tener cuatro demostraciones distintas")
    rotated = keys[ordinal % 4 :] + keys[: ordinal % 4]
    selected = [examples[(key["id"], key["version"])] for key in rotated[:shots]]
    if any(e["codigo_oa"] != row["codigo_oa"] for e in selected):
        raise ValueError("Demostración de otro OA")
    return conversation(row, selected), {"demonstrations": rotated[:shots]}
