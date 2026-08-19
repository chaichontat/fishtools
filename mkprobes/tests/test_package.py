import ast
import importlib
import pkgutil
from importlib.resources import files
from pathlib import Path

import mkprobes


def test_all_shipped_modules_import() -> None:
    modules = [module.name for module in pkgutil.walk_packages(mkprobes.__path__, "mkprobes.")]

    for module in modules:
        importlib.import_module(module)


def test_runtime_resources_are_packaged() -> None:
    resources = (
        files("mkprobes.codebook").joinpath("readout_ref_filtered.csv"),
        files("mkprobes.ext").joinpath("humanurls.tsv"),
        files("mkprobes.ext").joinpath("mouseurls.tsv"),
    )

    assert all(resource.is_file() for resource in resources)
    assert all(resource.read_text().strip() for resource in resources)


def test_source_does_not_import_fishtools() -> None:
    package_root = Path(mkprobes.__file__).parent
    forbidden: list[tuple[Path, int]] = []

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                name.name == "fishtools" or name.name.startswith("fishtools.") for name in node.names
            ):
                forbidden.append((path, node.lineno))
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "fishtools" or node.module.startswith("fishtools."):
                    forbidden.append((path, node.lineno))

    assert forbidden == []
