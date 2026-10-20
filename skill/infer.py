from common.prompts import conversation


def messages(row, *, instructions, **_):
    if (
        not instructions.startswith("---\n")
        or "name:" not in instructions
        or "description:" not in instructions
    ):
        raise ValueError("SKILL.md debe incluir metadatos name y description")
    return conversation(row, instructions=instructions), {"skill_active": True}
