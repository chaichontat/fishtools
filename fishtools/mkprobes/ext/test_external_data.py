import gzip
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Generator

import polars as pl
import pytest

from fishtools.mkprobes.ext.external_data import ExternalData, MockGTF


@pytest.fixture
def dummy_gtf_content() -> str:
    return """\
#comment line
chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; level 2; hgnc_id "HGNC:37102"; havana_gene "OTTHUMG00000000961.2";
chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; transcript_id "ENST00000456328.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "processed_transcript"; transcript_name "DDX11L1-202"; level 2; transcript_support_level "1"; hgnc_id "HGNC:37102"; tag "basic"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000362751.1";
chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id "ENSG00000223972.5"; transcript_id "ENST00000456328.2"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_type "processed_transcript"; transcript_name "DDX11L1-202"; exon_number 1; exon_id "ENSE00002234944.1"; level 2; transcript_support_level "1"; hgnc_id "HGNC:37102"; tag "basic"; havana_gene "OTTHUMG00000000961.2"; havana_transcript "OTTHUMT00000362751.1";
chr1\tHAVANA\tgene\t14404\t29570\t.\t+\t.\tgene_id "ENSG00000227232.5"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; level 2; hgnc_id "HGNC:38034"; havana_gene "OTTHUMG00000000958.1";
chr1\tHAVANA\ttranscript\t14404\t29570\t.\t+\t.\tgene_id "ENSG00000227232.5"; transcript_id "ENST00000488147.1"; gene_type "unprocessed_pseudogene"; gene_name "WASH7P"; transcript_type "unprocessed_pseudogene"; transcript_name "WASH7P-201"; level 2; transcript_support_level "NA"; hgnc_id "HGNC:38034"; tag "basic"; havana_gene "OTTHUMG00000000958.1"; havana_transcript "OTTHUMT00000003223.1";
"""


@pytest.fixture
def dummy_fasta_content() -> str:
    return """\
>ENST00000456328.2 cdna chromosome:GRCh38:1:11869:14409:1 gene:ENSG00000223972.5 gene_biotype:transcribed_unprocessed_pseudogene transcript_biotype:processed_transcript gene_symbol:DDX11L1 description:DEAD/H-box helicase 11 like 1 [Source:HGNC Symbol;Acc:HGNC:37102]
AGCTAGCTAGCTAGCTAGCT
>ENST00000488147.1 cdna chromosome:GRCh38:1:14404:29570:1 gene:ENSG00000227232.5 gene_biotype:unprocessed_pseudogene transcript_biotype:unprocessed_pseudogene gene_symbol:WASH7P description:WAS protein family homolog 7 pseudogene [Source:HGNC Symbol;Acc:HGNC:38034]
CGATCGATCGATCGATCGAT
"""


@pytest.fixture
def gtf_file(dummy_gtf_content: str) -> Generator[str, None, None]:
    # Create a temporary file. It's opened, written to, and then its path is yielded.
    # delete=False ensures the file isn't deleted when tmp_file_obj is closed.
    # We manually unlink it after the test.
    tmp_file_obj = tempfile.NamedTemporaryFile(mode="w+t", suffix=".gtf", delete=False, encoding="utf-8")
    with tmp_file_obj:
        tmp_file_obj.write(dummy_gtf_content)
        tmp_file_obj.flush()  # Ensure content is written to disk
        tmp_file_path = tmp_file_obj.name

    yield tmp_file_path
    os.unlink(tmp_file_path)


@pytest.fixture
def gtf_gz_file(dummy_gtf_content: str) -> Generator[str, None, None]:
    tmp_file_obj = tempfile.NamedTemporaryFile(mode="wb", suffix=".gtf.gz", delete=False)
    with tmp_file_obj:
        with gzip.open(tmp_file_obj, "wt", encoding="utf-8") as gz_writer:
            gz_writer.write(dummy_gtf_content)
        tmp_file_path = tmp_file_obj.name

    yield tmp_file_path
    os.unlink(tmp_file_path)


@pytest.fixture
def fasta_file(dummy_fasta_content: str) -> Generator[str, None, None]:
    tmp_file_obj = tempfile.NamedTemporaryFile(mode="w+t", suffix=".fasta", delete=False, encoding="utf-8")
    with tmp_file_obj:
        tmp_file_obj.write(dummy_fasta_content)
        tmp_file_obj.flush()
        tmp_file_path = tmp_file_obj.name

    yield tmp_file_path
    os.unlink(tmp_file_path)


@pytest.fixture
def cache_file(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.parquet"


@pytest.fixture
def ext_data_parsed(gtf_file: str, fasta_file: str, cache_file: Path) -> ExternalData:
    if cache_file.exists():
        cache_file.unlink()
    return ExternalData(cache=cache_file, gtf_path=gtf_file, fasta=fasta_file, regen_cache=True)


class TestExternalData:
    def test_parse_gtf_plain(self, gtf_file: str) -> None:
        df = ExternalData.parse_gtf(gtf_file)
        assert isinstance(df, pl.DataFrame)
        assert "gene_id" in df.columns
        assert "transcript_id" in df.columns
        assert "gene_name" in df.columns
        assert len(df) == 2
        assert df.filter(pl.col("gene_name") == "DDX11L1").height == 1
        assert df.filter(pl.col("transcript_id") == "ENST00000456328").height == 1

    def test_parse_gtf_gz(self, gtf_gz_file: str) -> None:
        df = ExternalData.parse_gtf(gtf_gz_file)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert df.filter(pl.col("gene_name") == "WASH7P").height == 1
        assert df.filter(pl.col("transcript_id") == "ENST00000488147").height == 1

    def test_parse_gtf_stringio(self, dummy_gtf_content: str) -> None:
        gtf_io = StringIO(dummy_gtf_content)
        df = ExternalData.parse_gtf(gtf_io)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_init_with_gtf_and_fasta(self, gtf_file: str, fasta_file: str, cache_file: Path) -> None:
        if cache_file.exists():
            cache_file.unlink()
        ext_data = ExternalData(cache=cache_file, gtf_path=gtf_file, fasta=fasta_file)
        assert isinstance(ext_data.gtf, pl.DataFrame)
        assert cache_file.exists()
        assert ext_data.fa["ENST00000456328"].seq == "AGCTAGCTAGCTAGCTAGCT"

    def test_init_with_cache(self, gtf_file: str, fasta_file: str, cache_file: Path) -> None:
        ExternalData(cache=cache_file, gtf_path=gtf_file, fasta=fasta_file, regen_cache=True)
        assert cache_file.exists()

        ext_data_cached = ExternalData(cache=cache_file, fasta=fasta_file)
        assert isinstance(ext_data_cached.gtf, pl.DataFrame)
        assert ext_data_cached.gtf.height == 2
        assert ext_data_cached.fa["ENST00000488147"].seq == "CGATCGATCGATCGATCGAT"

    def test_init_no_gtf_path_no_cache(self, fasta_file: str, cache_file: Path) -> None:
        if cache_file.exists():
            cache_file.unlink()
        ext_data = ExternalData(cache=cache_file, fasta=fasta_file)
        assert isinstance(ext_data.gtf, MockGTF)
        with pytest.raises(NotImplementedError):
            ext_data.gtf.some_method()  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError):
            _ = ext_data.gtf["some_key"]  # type: ignore[assignment]

    def test_gene_info(self, ext_data_parsed: ExternalData) -> None:
        info = ext_data_parsed.gene_info("DDX11L1")
        assert isinstance(info, pl.DataFrame)
        assert info.height == 1
        assert info[0, "gene_id"] == "ENSG00000223972"

    def test_gene_to_eid(self, ext_data_parsed: ExternalData) -> None:
        eid = ext_data_parsed.gene_to_eid("WASH7P")
        assert isinstance(eid, pl.Series)
        assert eid[0] == "ENSG00000227232"
        with pytest.raises(ValueError, match="Could not find NONEXISTENTGENE"):
            ext_data_parsed.gene_to_eid("NONEXISTENTGENE")

    def test_ts_to_gene(self, ext_data_parsed: ExternalData) -> None:
        gene_name = ext_data_parsed.ts_to_gene("ENST00000456328.2")
        assert gene_name == "DDX11L1"
        gene_name_no_version = ext_data_parsed.ts_to_gene("ENST00000456328")
        assert gene_name_no_version == "DDX11L1"
        assert ext_data_parsed.ts_to_gene("NONEXISTENT_TS") == "NONEXISTENT_TS"

    def test_ts_to_tsname(self, ext_data_parsed: ExternalData) -> None:
        ts_name = ext_data_parsed.ts_to_tsname("ENST00000488147.1")
        assert ts_name == "WASH7P-201"
        ts_name_no_version = ext_data_parsed.ts_to_tsname("ENST00000488147")
        assert ts_name_no_version == "WASH7P-201"
        assert ext_data_parsed.ts_to_tsname("NONEXISTENT_TS") is None

    def test_eid_to_ts(self, ext_data_parsed: ExternalData) -> None:
        ts_id = ext_data_parsed.eid_to_ts("ENSG00000223972.5")
        assert ts_id == "ENST00000456328"
        ts_id_no_version = ext_data_parsed.eid_to_ts("ENSG00000223972")
        assert ts_id_no_version == "ENST00000456328"

    def test_batch_convert(self, ext_data_parsed: ExternalData) -> None:
        src_vals = ["DDX11L1", "WASH7P", "NONEXISTENT"]
        df_converted = ext_data_parsed.batch_convert(src_vals, "gene_name", "gene_id")
        assert isinstance(df_converted, pl.DataFrame)
        assert df_converted.shape == (2, 2)
        assert "gene_name" in df_converted.columns
        assert "gene_id" in df_converted.columns
        assert df_converted.filter(pl.col("gene_name") == "DDX11L1")[0, "gene_id"] == "ENSG00000223972"
        assert df_converted.filter(pl.col("gene_name") == "NONEXISTENT").is_empty()

        with pytest.raises(ValueError, match="Could not find"):
            ext_data_parsed.batch_convert(["NONEXISTENT1", "NONEXISTENT2"], "gene_name", "gene_id")

    def test_convert(self, ext_data_parsed: ExternalData) -> None:
        gene_id = ext_data_parsed.convert("DDX11L1", "gene_name", "gene_id")
        assert gene_id == "ENSG00000223972"
        with pytest.raises(ValueError, match="Could not find NONEXISTENTGENE"):
            ext_data_parsed.convert("NONEXISTENTGENE", "gene_name", "gene_id")

    def test_get_transcripts_by_gene_name(self, ext_data_parsed: ExternalData) -> None:
        transcripts = ext_data_parsed.get_transcripts(gene="DDX11L1")
        assert isinstance(transcripts, pl.Series)
        assert len(transcripts) == 1
        assert transcripts[0] == "ENST00000456328"

    def test_get_transcripts_by_eid(self, ext_data_parsed: ExternalData) -> None:
        transcripts = ext_data_parsed.get_transcripts(eid="ENSG00000227232")
        assert isinstance(transcripts, pl.Series)
        assert len(transcripts) == 1
        assert transcripts[0] == "ENST00000488147"

    def test_get_seq(self, ext_data_parsed: ExternalData) -> None:
        seq = ext_data_parsed.get_seq("ENST00000456328.2")
        assert seq == "AGCTAGCTAGCTAGCTAGCT"
        seq_no_version = ext_data_parsed.get_seq("ENST00000456328")
        assert seq_no_version == "AGCTAGCTAGCTAGCTAGCT"
        with pytest.raises(ValueError, match="Could not find NONEXISTENT_TS"):  # pyfastx raises KeyError
            ext_data_parsed.get_seq("NONEXISTENT_TS")

    def test_get_seq_with_convert_transcript_name(self, ext_data_parsed: ExternalData) -> None:
        seq = ext_data_parsed.get_seq("DDX11L1-202", convert=True)
        assert seq == "AGCTAGCTAGCTAGCTAGCT"

    def test_filter_gene(self, ext_data_parsed: ExternalData) -> None:
        df_filtered = ext_data_parsed.filter_gene("WASH7P")
        assert isinstance(df_filtered, pl.DataFrame)
        assert df_filtered.height == 1
        assert df_filtered[0, "gene_id"] == "ENSG00000227232"

    def test_getitem_str_selects_column(self, ext_data_parsed: ExternalData) -> None:
        gene_names_series = ext_data_parsed["gene_name"]
        assert isinstance(gene_names_series, pl.Series)
        assert "DDX11L1" in gene_names_series.to_list()
        assert "WASH7P" in gene_names_series.to_list()

    def test_getitem_list_selects_columns(self, ext_data_parsed: ExternalData) -> None:
        selected_cols_df = ext_data_parsed[["gene_name", "transcript_id"]]
        assert isinstance(selected_cols_df, pl.DataFrame)
        assert "gene_name" in selected_cols_df.columns
        assert "transcript_id" in selected_cols_df.columns

    def test_parse_gtf_dynamic_attr_keys(
        self, dummy_gtf_content: str
    ) -> None:  # Changed to use dummy content directly
        gtf_content = """\
chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; custom_attr "Value1"; tag "basic";
chr1\tTEST\ttranscript\t200\t300\t.\t+\t.\tgene_id "G2"; transcript_id "T2"; another_attr "Value2"; level 2;
chr1\tTEST\ttranscript\t400\t500\t.\t+\t.\tgene_id "G3"; transcript_id "T3"; custom_attr "Value3"; yet_another "Hi";
"""
        # Use StringIO for this specific test as it's about parsing content
        df = ExternalData.parse_gtf(StringIO(gtf_content * 10))  # Repeat for sampling
        expected_columns = {
            "gene_id",
            "transcript_id",
            "custom_attr",
            "another_attr",
            "tag",
            "level",
            "yet_another",
        }
        for col in expected_columns:
            assert col in df.columns, f"Column {col} missing"

        assert df.filter(pl.col("transcript_id") == "T1")[0, "custom_attr"] == "Value1"
        assert df.filter(pl.col("transcript_id") == "T2")[0, "another_attr"] == "Value2"
        assert df.filter(pl.col("transcript_id") == "T3")[0, "yet_another"] == "Hi"
        assert df.filter(pl.col("custom_attr") == "Value1")[0, "gene_id"] == "G1"
        assert df.filter(pl.col("custom_attr") == "Value1")[0, "transcript_id"] == "T1"

    def test_id_extraction_in_parse_gtf(self) -> None:  # Changed to use StringIO
        gtf_content = """\
chrX\tHAVANA\ttranscript\t1\t1000\t.\t-\t.\tgene_id "ENSG001.10"; transcript_id "ENST001.5"; gene_name "TestGene";
chrX\tHAVANA\ttranscript\t2000\t3000\t.\t+\t.\tgene_id "ENSG002"; transcript_id "ENST002"; gene_name "AnotherGene";
"""
        df = ExternalData.parse_gtf(StringIO(gtf_content))

        assert df.filter(pl.col("gene_name") == "TestGene")[0, "gene_id"] == "ENSG001"
        assert df.filter(pl.col("gene_name") == "TestGene")[0, "transcript_id"] == "ENST001"
        assert df.filter(pl.col("gene_name") == "AnotherGene")[0, "gene_id"] == "ENSG002"
        assert df.filter(pl.col("gene_name") == "AnotherGene")[0, "transcript_id"] == "ENST002"

    def test_parse_gtf_filters_features(self) -> None:  # Changed to use StringIO
        gtf_content = """\
chr1\tTEST\tgene\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "T3";
chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "T1";
chr1\tTEST\texon\t10\t50\t.\t+\t.\tgene_id "G1"; transcript_id "T1";
chr1\tTEST\ttranscript\t200\t300\t.\t+\t.\tgene_id "G2"; transcript_id "T2";
"""
        gtf_io = StringIO(gtf_content)

        # Rewind StringIO buffer before each parse if using the same object
        gtf_io.seek(0)
        df_default = ExternalData.parse_gtf(gtf_io)
        assert df_default["feature"].unique().to_list() == ["transcript"]
        assert len(df_default) == 2

        gtf_io.seek(0)
        df_gene = ExternalData.parse_gtf(gtf_io, filters=("gene",))
        assert df_gene["feature"].unique().to_list() == ["gene"]
        assert len(df_gene) == 1

        gtf_io.seek(0)
        df_all_gtf_features = ExternalData.parse_gtf(gtf_io, filters=None)
        assert sorted(df_all_gtf_features["feature"].unique().to_list()) == sorted([
            "gene",
            "transcript",
            "exon",
        ])
        assert len(df_all_gtf_features) == 4

    def test_parse_gtf_missing_required_ids(self) -> None:
        # Test case where gene_id is missing
        gtf_content_no_gene_id = """\
chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\ttranscript_id "T1"; gene_name "TestGene1";
"""
        with pytest.raises(
            ValueError, match="Gene ID not found in GTF file. Required attribute per GTF 2.0 spec."
        ):
            ExternalData.parse_gtf(StringIO(gtf_content_no_gene_id))

        # Test case where transcript_id is missing
        gtf_content_no_transcript_id = """\
chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; gene_name "TestGene2";
"""
        with pytest.raises(
            ValueError, match="Transcript ID not found in GTF file. Required attribute per GTF 2.0 spec."
        ):
            ExternalData.parse_gtf(StringIO(gtf_content_no_transcript_id))

        # Test case where both are present (should not raise error related to missing keys)
        gtf_content_both_present = """\
chr1\tTEST\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; gene_name "TestGene3";
"""
        try:
            ExternalData.parse_gtf(StringIO(gtf_content_both_present))
        except ValueError as e:
            # We only want to fail if it's one of the specific key missing errors
            assert "Gene ID not found" not in str(e)
            assert "Transcript ID not found" not in str(e)
            # If another ValueError occurs, this test isn't designed to catch it,
            # but it shouldn't be due to missing gene_id/transcript_id.
            # Depending on strictness, one might re-raise or pass.
            # For this test, we assume other ValueErrors are not the target.
            pass


class TestExternalDataNoGTF:
    """
    Tests for ExternalData when no GTF path is provided and no cache exists.
    In this scenario, self.gtf should be a MockGTF instance.
    """

    @pytest.fixture
    def ext_data_no_gtf(self, fasta_file: str, cache_file: Path) -> ExternalData:
        """
        Provides an ExternalData instance initialized without a GTF file
        and with no pre-existing cache.
        """
        if cache_file.exists():
            cache_file.unlink()
        return ExternalData(cache=cache_file, fasta=fasta_file, gtf_path=None)

    def test_init_results_in_mock_gtf(self, ext_data_no_gtf: ExternalData) -> None:
        assert isinstance(ext_data_no_gtf.gtf, MockGTF)
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.gtf.filter(pl.col("any") == "any")  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError):
            _ = ext_data_no_gtf.gtf["some_column"]  # type: ignore[assignment]

    def test_gene_info_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.gene_info("any_gene")

    def test_gene_to_eid_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.gene_to_eid("any_gene")

    def test_ts_to_gene_returns_input(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.ts_to_gene("any_ts.1")

    def test_ts_to_tsname_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.ts_to_tsname("any_ts_id")

    def test_eid_to_ts_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.eid_to_ts("any_gene_id")

    def test_batch_convert_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.batch_convert(["val1"], "src_col", "dst_col")

    def test_convert_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.convert("val1", "src_col", "dst_col")

    def test_get_transcripts_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.get_transcripts(gene="any_gene")
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.get_transcripts(eid="any_eid")

    def test_get_seq_behavior_no_gtf(self, ext_data_no_gtf: ExternalData, dummy_fasta_content: str) -> None:
        # Test with a valid FASTA ID, convert=False
        # The dummy_fasta_content has "ENST00000456328.2" which becomes "ENST00000456328" as key
        assert ext_data_no_gtf.get_seq("ENST00000456328", convert=False) == "AGCTAGCTAGCTAGCTAGCT"

        # Test with convert=True, which would call self.convert if a transcript name is passed
        # self.convert should raise NotImplementedError due to MockGTF
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.get_seq("Some-Transcript-Name", convert=True)

        # Test with a valid FASTA ID that doesn't look like a transcript name, convert=True
        # Should bypass the convert call and directly access FASTA
        assert ext_data_no_gtf.get_seq("ENST00000456328", convert=True) == "AGCTAGCTAGCTAGCTAGCT"

        with pytest.raises(ValueError, match="Could not find NONEXISTENT_TS in fasta file."):
            ext_data_no_gtf.get_seq("NONEXISTENT_TS", convert=False)

    def test_filter_gene_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            ext_data_no_gtf.filter_gene("any_gene")

    def test_getitem_raises_notimplemented(self, ext_data_no_gtf: ExternalData) -> None:
        with pytest.raises(NotImplementedError):
            _ = ext_data_no_gtf["some_column"]  # type: ignore[assignment]
        with pytest.raises(NotImplementedError):
            _ = ext_data_no_gtf[["col1", "col2"]]  # type: ignore[assignment]
