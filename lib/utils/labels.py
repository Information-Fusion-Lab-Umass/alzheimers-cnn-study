LABELS = ["CN", "MCI", "AD"]


def encode_label(label: str) -> int:
    return LABELS.index(label)


def decode_label(index: int) -> str:
    return LABELS[index]
