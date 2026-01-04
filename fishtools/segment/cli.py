from __future__ import annotations

import logging
from typing import Any

import rich_click as click


class SegmentCLI(click.Group):
    """Click group that surfaces exceptions (standalone_mode=False by default)."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("standalone_mode", False)
        return super().main(*args, **kwargs)


app = SegmentCLI(help="Segmentation tooling CLI.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app()

