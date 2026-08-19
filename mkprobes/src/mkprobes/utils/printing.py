import json
from typing import Any

from rich.console import Console
from rich.syntax import Syntax

console = Console()


def jprint(value: Any) -> None:
    console.print(Syntax(json.dumps(value, indent=2), "json"))

