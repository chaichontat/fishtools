from click.testing import CliRunner

from fishtools.preprocess.spots.align_prod import spots as spots_cli


def test_spots_investigate_help_smoke() -> None:
    runner = CliRunner()
    res = runner.invoke(spots_cli, ["investigate", "--help"])
    assert res.exit_code == 0, res.output
    assert "--peak-perc" in res.output

