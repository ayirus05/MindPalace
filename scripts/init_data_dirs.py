"""Create the runtime data directory structure for MindPalace."""

from pathlib import Path


DATA_LEAF_DIRS = (
    "index",
    "cache",
    "source_docs",
    "logs",
    "vault/system",
    "vault/user",
)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"

    for relative_dir in DATA_LEAF_DIRS:
        leaf_dir = data_dir / relative_dir
        leaf_dir.mkdir(parents=True, exist_ok=True)
        (leaf_dir / ".gitkeep").touch(exist_ok=True)


if __name__ == "__main__":
    main()
