from importlib import import_module
from types import SimpleNamespace

import rich_click as click

LAZY_COMMANDS: dict[str, SimpleNamespace] = {
    "export-mask-edit-pack": SimpleNamespace(module="fishtools.ccf.cli_export_mask_edit_pack", attr="main"),
    "export-moving-thumbnail": SimpleNamespace(module="fishtools.ccf.cli_export_moving_thumbnail", attr="main"),
    "export-atlas-slice": SimpleNamespace(module="fishtools.ccf.cli_export_atlas_slice", attr="main"),
    "warp-h5ad-spatial": SimpleNamespace(module="fishtools.ccf.cli_warp_h5ad_spatial", attr="main"),
    "filter-h5ad-ccf": SimpleNamespace(module="fishtools.ccf.cli_filter_h5ad_ccf", attr="main"),
    "princurve-qc-review": SimpleNamespace(module="fishtools.ccf.cli_princurve_qc_review", attr="main"),
    "princurve-signed-r-review": SimpleNamespace(module="fishtools.ccf.cli_princurve_signed_r_review", attr="main"),
    "princurve-signed-r-review-all": SimpleNamespace(
        module="fishtools.ccf.cli_princurve_signed_r_review_all", attr="main"
    ),
}


class LazyGroup(click.Group):
    """Lazy-loading Click group that defers CLI imports until invocation."""

    def __init__(
        self, *args: object, lazy_commands: dict[str, SimpleNamespace] | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        eager = super().list_commands(ctx)
        lazy = sorted(self._lazy_commands)
        ordered = list(dict.fromkeys([*eager, *lazy]))
        return ordered

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None

        module = import_module(spec.module)
        return getattr(module, spec.attr)


click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
def main() -> None:
    """CCF workflow utilities (register → mask edit → warp → label/filter).

    This is an umbrella CLI that groups the CCF commands described in `docs/ccf-process.md`.
    """


if __name__ == "__main__":
    main()
