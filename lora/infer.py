from common.prompts import conversation


def messages(row, **_):
    return conversation(row), {"adapter_active": True}
