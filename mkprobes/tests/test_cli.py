import json
from pathlib import Path

from click.testing import CliRunner

from mkprobes import cli
from mkprobes.ext import prepare as prepare_module


def test_cli_help_lists_probe_workflow_commands() -> None:
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "candidates" in result.output
    assert "screen" in result.output
    assert "construct" in result.output
    assert "prepare" in result.output


def test_hash_prints_codebook_hash(tmp_path: Path) -> None:
    codebook = tmp_path / "codebook.json"
    codebook.write_text(json.dumps({"GeneA": [1, 2]}))

    result = CliRunner().invoke(cli.main, ["hash", str(codebook)])

    assert result.exit_code == 0
    assert result.output.strip() == "0f5169"


def test_prepare_orchestrates_reference_build(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(path: Path, species: str) -> None:
        calls.append((f"download:{species}", path))

    def fake_jellyfish(path: Path) -> None:
        calls.append(("jellyfish", path))

    def fake_bowtie(fasta: Path, name: str) -> None:
        calls.append((f"bowtie:{name}", fasta))

    def fake_dataset(path: Path) -> None:
        calls.append(("dataset", path))

    monkeypatch.setattr(prepare_module, "download_gtf_fasta", fake_download)
    monkeypatch.setattr(prepare_module, "run_jellyfish", fake_jellyfish)
    monkeypatch.setattr(cli, "bowtie_build", fake_bowtie)
    monkeypatch.setattr(cli, "Dataset", fake_dataset)

    result = CliRunner().invoke(cli.main, ["prepare", str(tmp_path), "--species", "mouse"])

    reference = tmp_path.resolve() / "mouse"
    assert result.exit_code == 0
    assert set(calls) == {
        ("download:mouse", reference),
        ("jellyfish", reference),
        ("bowtie:txome", reference / "cdna_ncrna_trna.fasta"),
        ("dataset", reference),
    }
