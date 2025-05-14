import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fishtools.mkprobes.ext.dataset import (
    Dataset,
    DatasetDefinition,
    ReferenceDataset,
    parse_jellyfish,
)
from fishtools.mkprobes.ext.external_data import (
    ExternalData,
    ExternalDataDefinition,
    MockGTF,
)


@pytest.fixture
def dummy_fasta_content() -> str:
    return ">seq1\nAGCTAGCT\n>seq2\nTCGATCGA\n"


@pytest.fixture
def dummy_jellyfish_content() -> str:
    return "AGCTAGCTAGCTAGCTAG 2\nTCGATCGATCGATCGATC 3\n"


@pytest.fixture
def dummy_appris_content() -> str:
    return "GENE_A\tENSG00A\tENST00A\tCCDS1\tPRINCIPAL:1\nGENE_B\tENSG00B\tENST00B\tCCDS2\tALTERNATIVE:2\n"


@pytest.fixture
def dummy_gtf_content() -> str:
    return """
# comment
chr1\tHAVANA\tgene\t1\t100\t.\t+\t.\tgene_id "ENSG00000223972"; gene_name "DDX11L1";
chr1\tHAVANA\ttranscript\t1\t100\t.\t+\t.\tgene_id "ENSG00000223972"; transcript_id "ENST00000456328"; gene_name "DDX11L1";
"""


@pytest.fixture
def mock_external_data(tmp_path: Path, dummy_fasta_content: str) -> ExternalData:
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(dummy_fasta_content)

    # Mock methods that might involve heavy computation or external calls
    mock_ed = MagicMock(spec=ExternalData)
    mock_ed.fasta_path = fasta_file
    mock_ed.fa = MagicMock()  # Simplified mock for pyfastx.Fasta
    mock_ed.gtf = MockGTF()
    mock_ed.bowtie2_index = tmp_path / "test_bowtie_index"
    mock_ed.kmer = tmp_path / "test_kmer.jf"
    mock_ed.bowtie_build = MagicMock()
    mock_ed.run_jellyfish = MagicMock()
    return mock_ed


class TestParseJellyfish:
    def test_parse_valid_file(self, tmp_path: Path, dummy_jellyfish_content: str):
        jf_file = tmp_path / "test.jf"
        jf_file.write_text(dummy_jellyfish_content)
        df = parse_jellyfish(jf_file)
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["kmer", "count"]
        assert len(df) == 2
        assert df["kmer"][0] == "AGCTAGCTAGCTAGCTAG"
        assert df["count"][1] == 3

    def test_parse_empty_file(self, tmp_path: Path):
        jf_file = tmp_path / "empty.jf"
        jf_file.write_text("")
        with pytest.raises(ValueError, match="Jellyfish file"):
            parse_jellyfish(jf_file)

    def test_parse_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_jellyfish("non_existent_file.jf")


class TestDataset:
    def test_init_minimal(self, tmp_path: Path, mock_external_data: MagicMock):
        dataset = Dataset(path=tmp_path, external_data=mock_external_data, species="test_species")
        assert dataset.path == tmp_path
        assert dataset.data == mock_external_data
        assert dataset.species == "test_species"
        assert dataset.kmer18 is None
        assert not dataset.kmerset
        assert dataset.trna_rna_kmers is None
        assert dataset.gencode == mock_external_data  # Backwards compatibility
        assert dataset.ensembl is None

    def test_init_with_kmers(
        self, tmp_path: Path, mock_external_data: MagicMock, dummy_jellyfish_content: str
    ):
        kmer18_file = tmp_path / "k18.jf"
        kmer18_file.write_text(dummy_jellyfish_content)
        trna_kmers_file = tmp_path / "trna.jf"
        trna_kmers_file.write_text("CCCC 1\n")

        dataset = Dataset(
            path=tmp_path,
            external_data=mock_external_data,
            kmer18_path=kmer18_file,
            trna_rna_kmers_path=trna_kmers_file,
        )
        assert dataset.kmer18 is not None
        assert len(dataset.kmer18) == 2
        assert "AGCTAGCTAGCTAGCTAG" in dataset.kmerset
        assert dataset.trna_rna_kmers is not None
        assert "CCCC" in dataset.trna_rna_kmers

    @patch("fishtools.mkprobes.ext.dataset.ExternalData.from_definition")
    def test_from_folder(self, mock_ed_from_def: MagicMock, tmp_path: Path):
        dataset_path = tmp_path / "my_dataset_folder"
        dataset_path.mkdir()

        mock_ed_instance = MagicMock(spec=ExternalData)
        mock_ed_instance.kmer = "test.jf"  # kmer path for constructor
        mock_ed_from_def.return_value = mock_ed_instance

        definition_content = {
            "species": "folder_species",
            "external_data": {
                "default": {
                    "fasta_name": "test.fa",
                    "bowtie2_index_name": "test.bt2",
                    "kmer18_name": "test.jf",
                }
            },
        }
        (dataset_path / "dataset.json").write_text(json.dumps(definition_content))
        (dataset_path / "test.jf").write_text("AGCTAGCTAGCTAGCTAG 2\nTCGATCGATCGATCGATC 3\n")

        dataset = Dataset.from_folder(dataset_path)

        expected_def = ExternalDataDefinition(**definition_content["external_data"]["default"])
        mock_ed_from_def.assert_called_once_with(dataset_path, expected_def)
        assert dataset.species == "folder_species"
        assert dataset.data == mock_ed_instance
        assert dataset.path == dataset_path

    def test_from_folder_no_json(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist. Please create a dataset first."):
            Dataset.from_folder(tmp_path / "non_existent_dataset_dir")

    def test_check_kmers(self, tmp_path: Path, mock_external_data: MagicMock):
        dataset = Dataset(path=tmp_path, external_data=mock_external_data)

        # Case 1: trna_rna_kmers is None
        assert not dataset.check_kmers("AGCT")

        # Case 2: trna_rna_kmers is set
        dataset.trna_rna_kmers = {"AGCT", "GGGG"}  # Assuming kmer length 4 for simplicity here
        with patch("fishtools.mkprobes.ext.dataset.kmers", return_value=["AGCT", "GCTA", "CTAG"]):
            assert dataset.check_kmers("AGCTAG")
        with patch("fishtools.mkprobes.ext.dataset.kmers", return_value=["TTTT", "AAAA"]):
            assert not dataset.check_kmers("TTTTAA")

    def test_appris_property_not_implemented(self, tmp_path: Path, mock_external_data: MagicMock):
        dataset = Dataset(path=tmp_path, external_data=mock_external_data)
        with pytest.raises(NotImplementedError):
            _ = dataset.appris


class TestDatasetFromComponents:
    @patch("fishtools.mkprobes.ext.dataset.ExternalData")
    def test_from_components_success(
        self,
        mock_external_data_cls: MagicMock,
        tmp_path: Path,
        dummy_fasta_content: str,  # Make sure this fixture provides non-empty FASTA content
        dummy_jellyfish_content: str,  # Make sure this fixture provides non-empty k-mer content
    ):
        # 1. Setup paths and source FASTA file
        source_fasta_dir = tmp_path / "source_files"
        source_fasta_dir.mkdir()
        source_fasta_file = source_fasta_dir / "input.fasta"
        source_fasta_file.write_text(dummy_fasta_content)
        assert source_fasta_file.read_text() != "", "Dummy FASTA content should not be empty"

        dataset_path = tmp_path / "my_dataset"
        species = "test_organism"

        # 2. Configure the mock ExternalData instance
        mock_ed_instance = MagicMock(spec=ExternalData)

        # ExternalData is initialized with the original fasta_file path
        mock_ed_instance.fasta_path = source_fasta_file

        # Mock properties that return paths/names based on the original fasta_file
        # .kmer property will return source_fasta_file.with_suffix(".jf")
        expected_kmer_file_path = source_fasta_file.with_suffix(".jf")
        mock_ed_instance.kmer = expected_kmer_file_path

        # .bowtie2_index property will return source_fasta_file.with_suffix("") (the stem)
        expected_bowtie_index_stem_path = source_fasta_file.with_suffix("")
        mock_ed_instance.bowtie2_index = expected_bowtie_index_stem_path

        # Mock methods
        mock_ed_instance.bowtie_build = MagicMock()
        mock_ed_instance.run_jellyfish = MagicMock()

        mock_external_data_cls.return_value = mock_ed_instance

        # 3. Create the dummy k-mer file that Dataset.__init__ will parse.
        # This file is located relative to the *source* FASTA, as per external_data.kmer.
        expected_kmer_file_path.write_text(dummy_jellyfish_content)
        assert expected_kmer_file_path.read_text() != "", "Dummy k-mer content should not be empty"

        # 4. Call Dataset.from_components
        dataset = Dataset.from_components(
            path=dataset_path, fasta_file=source_fasta_file, species=species, overwrite=True
        )

        # 5. Assertions
        # Assertions for ExternalData method calls
        mock_external_data_cls.assert_called_once_with(
            cache=dataset_path / source_fasta_file.with_suffix(".parquet").name,  # Cache is in dataset_path
            fasta=dataset_path / source_fasta_file.name,  # Initialized with original fasta_file
            regen_cache=True,
        )
        mock_ed_instance.bowtie_build.assert_called_once()
        mock_ed_instance.run_jellyfish.assert_called_once()

        # Assertion for file copying
        # shutil.copy is called if source and target are different.
        # Target is dataset_path / source_fasta_file.name
        expected_copied_fasta_path = dataset_path / source_fasta_file.name

        # Assertions for dataset.json
        dataset_json_path = dataset_path / "dataset.json"
        assert dataset_json_path.exists(), "dataset.json was not created"

        json_data = json.loads(dataset_json_path.read_text())

        assert json_data["species"] == species
        assert "default" in json_data["external_data"]
        default_ext_data_def = json_data["external_data"]["default"]

        assert default_ext_data_def["fasta_name"] == source_fasta_file.name
        assert default_ext_data_def["bowtie2_index_name"] == expected_bowtie_index_stem_path.name
        assert default_ext_data_def["kmer18_name"] == expected_kmer_file_path.name

        # Assertions for the returned Dataset object
        assert isinstance(dataset, Dataset)
        assert dataset.path == dataset_path
        assert dataset.species == species
        assert dataset.data == mock_ed_instance
        assert dataset.kmer18 is not None, "Dataset kmer18 data should be loaded"
        assert len(dataset.kmer18) > 0, "Dataset kmer18 data should not be empty"
        assert "AGCTAGCTAGCTAGCTAG" in dataset.kmerset  # Based on dummy_jellyfish_content

        # Check that the FASTA file was indeed copied (even though shutil.copy is mocked,
        # the path used by ExternalData for its cache, etc., implies this structure)
        assert expected_copied_fasta_path.exists(), "FASTA file was not copied to dataset directory"
        assert expected_copied_fasta_path.read_text() == dummy_fasta_content
