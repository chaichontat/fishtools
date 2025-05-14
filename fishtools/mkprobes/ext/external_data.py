# %%
import gzip
import json
from collections.abc import Callable
from functools import cache
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Any, Sequence, overload

import polars as pl
import pyfastx
import requests
from loguru import logger
from pydantic import BaseModel

from fishtools.mkprobes.utils._alignment import bowtie_build, jellyfish


def get_ensembl(path: Path | str, id_: str):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    if (p := (path / f"{id_}.json")).exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Error decoding {p}. Deleting.")
            p.unlink(missing_ok=True)

    logger.info(f"Fetching {id_} on ensembl")
    res = requests.get(f"https://rest.ensembl.org/lookup/id/{id_}?content-type=application/json", timeout=30)
    res.raise_for_status()
    p.write_text(json.dumps(j := res.json(), indent=2))
    return j


class MockGTF:
    """
    A placeholder for GTF data when a GTF file is not provided or cannot be loaded.

    This class is used internally by `ExternalData` when GTF information is unavailable.
    Any attempt to access attributes or items that would normally come from a parsed
    GTF file will result in a `NotImplementedError`.
    """

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"ExternalData not created with GTF path. Cannot access {name}")

    def __getitem__(self, name: str | list[str]) -> Any:
        raise NotImplementedError("ExternalData not created with GTF path. Cannot index.")


class ExternalDataDefinition(BaseModel):
    """
    Defines the structure for storing external data source file names.

    This Pydantic model is used to serialize and deserialize the names of files
    associated with an `ExternalData` instance, such as the cache, GTF, FASTA,
    k-mer count file, and Bowtie2 index. It's primarily used when saving or
    loading dataset configurations that include `ExternalData`.

    Potential Pitfalls:
        - All file names are relative to a base path that is typically managed by
          the `Dataset` class. Ensure these names correctly point to existing files
          within that context when an `ExternalData` instance is created using
          this definition.
        - `cache_name` and `gtf_name` are optional. If `gtf_name` is not provided,
          the resulting `ExternalData` instance will use `MockGTF` for its GTF component,
          limiting its functionality.
    """

    cache_name: str | None = None
    gtf_name: str | None = None
    fasta_name: str
    kmer18_name: str
    bowtie2_index_name: str


class ExternalData:
    """
    Manages access to genomic data from GTF (Gene Transfer Format) and FASTA files.

    It provides methods to parse GTF files, retrieve gene and transcript information,
    and fetch sequences from FASTA files. The class uses caching for parsed GTF
    data to speed up subsequent initializations.

    Normal Use:
    -----------
    1. Initialize with paths to a cache location, a FASTA file, and a GTF file:
       ```python
       from pathlib import Path
       ext_data = ExternalData(
           cache="path/to/cache.parquet",  # Path for caching parsed GTF data
           fasta="path/to/genome.fasta",
           gtf_path="path/to/annotations.gtf"
       )
       ```
       The `cache` file does not need to exist beforehand; it will be created if a
       `gtf_path` is provided and the file is not already present or `regen_cache` is True.

    2. Access GTF data as a Polars DataFrame:
       ```python
       gtf_df = ext_data.gtf
       gene_info_df = ext_data.gene_info("MyGeneName")
       ```
    3. Retrieve sequences:
       ```python
       sequence = ext_data.get_seq("ENST00000123456")
       ```
    4. Convert between different ID types:
       ```python
       gene_id = ext_data.convert("MyGeneName", "gene_name", "gene_id")
       ```

    Potential Pitfalls:
    -------------------
    - **Caching**:
        - Parsed GTF data is cached to `cache` path. If the GTF file changes,
          `regen_cache=True` must be set during initialization to re-parse and
          update the cache. Otherwise, stale data might be used.
        - If `cache` path exists and `regen_cache=False` (default), `gtf_path` is
          ignored, and data is loaded directly from the cache.

    - **Missing GTF Path**:
        - If `gtf_path` is `None` AND no valid cache file exists at the `cache` path,
          `self.gtf` will be a `MockGTF` instance.
        - Most methods relying on GTF data (e.g., `gene_info`, `convert`, `get_transcripts`)
          will raise `NotImplementedError` or return default/input values.
        - This mode is intended for scenarios where only FASTA access is needed, or
          GTF data is managed externally.

    - **FASTA Key Function (`fasta_key_func`)**:
        - This function normalizes sequence headers from the FASTA file to generate
          keys for sequence lookup (e.g., extracting transcript IDs).
        - The default function captures the first word.
        - If your FASTA headers have a different format, you MUST provide a custom
          `fasta_key_func` to ensure IDs match those used/derived from the GTF.
          Mismatched keys will lead to `KeyError` or `ValueError` when fetching sequences.

    - **Required GTF Attributes**:
        - The `parse_gtf` method (and thus initialization with a `gtf_path`) expects
          `gene_id` and `transcript_id` to be present in the GTF attributes for each
          feature row after sampling.
        - If these are missing from the sampled attributes, a `ValueError` will be raised.
          Ensure your GTF file adheres to this or that the sampled rows are representative.

    - **GTF Attribute Discovery**:
        - Attribute keys (e.g., `gene_name`, `transcript_type`) are dynamically discovered
          by sampling a subset of rows from the GTF file (default sample size is 25).
        - If your GTF file is very large and has rare attribute keys not present in the
          sample, these keys might not be parsed into separate columns.

    - **ID Versioning**:
        - The `parse_gtf` method attempts to strip version suffixes (e.g., ".1", ".5")
          from `gene_id` and `transcript_id` by default.
        - Methods like `ts_to_gene`, `eid_to_ts`, and `get_seq` also often strip
          version suffixes from input IDs before lookup. Be mindful of this if your
          workflow relies on versioned IDs.
    """

    def __init__[T: str](
        self,
        cache: Path | str | None = None,
        *,
        fasta: Path | str,
        gtf_path: Path | str | None = None,
        regen_cache: bool = False,
        fasta_key_func: Callable[[T], T] = lambda x: x.split(" ")[0],
        bowtie2_index: str | None = None,
        kmer18: str | None = None,
    ) -> None:
        """
        Initializes an ExternalData object.

        See the class docstring for detailed information on parameters and behavior.

        Args:
            cache: Path to the Parquet file for caching parsed GTF data.
            fasta: Path to the FASTA file.
            gtf_path: Path to the GTF file. Optional if cache exists or only FASTA access is needed.
            regen_cache: If True, forces re-parsing of the GTF file and overwrites the cache.
            fasta_key_func: Function to extract a lookup key from FASTA headers.
            bowtie2_index: Optional explicit name for the Bowtie2 index files (stem).
            kmer18: Optional explicit name for the 18-mer Jellyfish output file.
        """
        self.fasta_path = Path(fasta).resolve()
        try:
            self.fa = pyfastx.Fasta(Path(fasta).as_posix(), key_func=fasta_key_func)
        except Exception as e:
            raise Exception(
                f"Error reading FASTA file {fasta}. Ensure the file is valid and not empty."
            ) from e

        self._ts_gene_map: dict[str, str] | None = None

        if cache and Path(cache).exists() and not regen_cache:
            self.gtf: pl.DataFrame | MockGTF = pl.read_parquet(cache)
        else:
            if gtf_path is None:
                # logger.warning("GTF path not specified. Must be specified for reference species.")
                self.gtf = MockGTF()
            else:
                if not cache:
                    raise ValueError("Cache path must be specified if GTF path is provided.")
                self.gtf = self.parse_gtf(Path(gtf_path).resolve())
                self.gtf.write_parquet(cache)

        self.key_func = key_func
        self._override_bowtie2_index = bowtie2_index
        self._override_kmer18 = kmer18

    @classmethod
    def from_definition(cls, path: Path, definition: ExternalDataDefinition):
        """
        Creates an ExternalData instance from an ExternalDataDefinition.
        """
        return cls(
            cache=path / definition.cache_name if definition.cache_name else None,
            fasta=path / definition.fasta_name,
            gtf_path=path / definition.gtf_name if definition.gtf_name else None,
            bowtie2_index=definition.bowtie2_index_name,
            kmer18=definition.kmer18_name,
        )

    @property
    def bowtie2_index(self):
        """
        Gets the base path for the Bowtie2 index files.

        If `bowtie2_index` was provided during class initialization, that is used.
        Otherwise, it searches for index files (e.g., `{self.fasta_path}.1.bt2`) in the
        same directory as the FASTA file.

        Returns:
            Path: The base path (stem) of the Bowtie2 index files.

        Raises:
            FileNotFoundError: If the Bowtie2 index cannot be found.
        """
        if self._override_bowtie2_index:
            return self.fasta_path.parent / self._override_bowtie2_index

        bt = self.fasta_path.parent.glob(f"{self.fasta_path.stem}*.bt2")
        if not len(list(bt)):
            raise FileNotFoundError(
                f"Bowtie2 index not found for {self.fasta_path.stem}. Please build with `bowtie2-build {{fasta_path}} {{fasta file name}}` or call self.bowtie_build()."
            )

        return self.fasta_path.with_suffix("")

    def bowtie_build(self, overwrite: bool = False):
        """
        Builds the Bowtie2 index for the FASTA file if it doesn't already exist.

        Checks for the existence of `fasta_stem.1.bt2`. If not found, it prompts
        the user before running `bowtie2-build`.

        Raises:
            FileNotFoundError: If `bowtie2-build` fails to create the index files.
        """
        if self.fasta_path.with_suffix(".1.bt2").exists() and not overwrite:
            return
        logger.info(f"Bowtie2 index not found for {self.fasta_path.stem}.")
        input("\nPress Enter to start building...")
        bowtie_build(self.fasta_path, self.fasta_path.stem)
        if not self.fasta_path.with_suffix(".1.bt2").exists():
            raise FileNotFoundError(
                f"Bowtie2 index not found for {self.fasta_path.stem}. Bowtie2 build failed."
            )
        logger.info(f"Bowtie2 index successfully built for {self.fasta_path.stem}.")

    @property
    def kmer(self):
        """
        Gets the path to the k-mer count file (Jellyfish output).

        If `kmer18` was provided, that is used. Otherwise, it looks for
        a file named `{self.fasta_stem}.jf` in the same directory as the FASTA file.

        Returns:
            Path: The path to the k-mer file.

        Raises:
            FileNotFoundError: If the k-mer file cannot be found.
        """
        if self._override_kmer18:
            kmer18 = self.fasta_path.parent / self._override_kmer18
            if not kmer18.exists():
                raise FileNotFoundError(
                    f"Kmer file {kmer18} not found. Please run jellyfish or recreate the dataset."
                )
            return self.fasta_path.parent / kmer18

        if not self.fasta_path.with_suffix(".jf").exists():
            raise FileNotFoundError("Kmer file not found. Please run jellyfish.")
        return self.fasta_path.with_suffix(".jf")

    def run_jellyfish(self, kmer: int = 18, overwrite: bool = False):
        """
        Runs Jellyfish to count k-mers from the sequences in the FASTA file.

        Generates a `.jf` file (e.g., `fasta_stem.jf`). If the output file
        already exists and `overwrite` is False, the operation is skipped.
        Prompts the user before running if not overwriting.

        Args:
            kmer: The k-mer size to use for counting. Defaults to 18.
            overwrite: If True, run Jellyfish even if the output file exists.
                Defaults to False.

        Raises:
            FileNotFoundError: If Jellyfish fails to create the output file.
        """
        if self.fasta_path.with_suffix(".jf").exists() and not overwrite:
            logger.info(f"Jellyfish file {self.fasta_path.with_suffix('.jf')} already exists. Skipping.")
            return

        logger.info("Need to run jellyfish to get 18-mers in cDNA.")
        if not overwrite:
            input("Press Enter to start running...")

        jellyfish(
            [x.seq for x in self.fa],
            self.fasta_path.with_suffix(".jf"),
            kmer,
            minimum=10,
            counter=4,
        )
        if not self.fasta_path.with_suffix(".jf").exists():
            raise FileNotFoundError("cdna18.jf not found. Jellyfish run failed.")
        logger.info(f"Jellyfish file {self.fasta_path.with_suffix('.jf')} successfully created.")

    @cache
    def gene_info(self, gene: str) -> pl.DataFrame:
        """
        Retrieves all GTF entries for a given gene name.

        Args:
            gene: The gene name (e.g., "Actb").

        Returns:
            A Polars DataFrame containing rows from the GTF that match the gene name.
            Returns an empty DataFrame if the gene is not found or if GTF data is unavailable.
        """
        return self.gtf.filter(pl.col("gene_name") == gene)

    @cache
    def gene_to_eid(self, gene: str) -> pl.Series:
        """
        Converts a gene name to its corresponding gene ID(s) (e.g., Ensembl ID).

        Args:
            gene: The gene name.

        Returns:
            A Polars Series containing the gene ID(s) for the given gene name.

        Raises:
            ValueError: If the gene name is not found in the GTF data.
        """
        ret = self.gene_info(gene)
        if ret.is_empty():
            raise ValueError(f"Could not find {gene}")
        return ret[:, "gene_id"]

    @cache
    def ts_to_gene(self, ts: str) -> str:
        """
        Maps a transcript ID (version stripped) to its corresponding gene name.

        Uses an internal cached mapping. If the transcript ID is not found,
        it returns the input transcript ID.

        Args:
            ts: The transcript ID (e.g., "ENSMUST00000000001.4" or "ENSMUST00000000001").

        Returns:
            The gene name associated with the transcript ID, or the input `ts` if not found.
        """
        ts = self.key_func(ts)
        if self._ts_gene_map is None:
            self._ts_gene_map = {k: v for k, v in zip(self.gtf["transcript_id"], self.gtf["gene_name"])}
        return self._ts_gene_map.get(ts, ts)

    @cache
    def ts_to_tsname(self, eid: str) -> str | None:
        """
        Converts a transcript ID (version stripped) to its transcript name.

        If the transcript ID is found, returns the 'transcript_name' attribute.
        Otherwise, returns the input transcript ID.

        Args:
            eid: The transcript ID (e.g., "ENSMUST00000000001.4" or "ENSMUST00000000001").

        Returns:
            The transcript name, or the input `eid` if not found or if 'transcript_name'
            is missing.
        """
        eid = eid.split(".")[0]
        try:
            return self.gtf.filter(pl.col("transcript_id") == eid)["transcript_name"].first()  # type: ignore
        except pl.exceptions.ComputeError:
            return eid

    @cache
    def eid_to_ts(self, eid: str) -> str:
        """
        Converts a gene ID (Ensembl ID, version stripped) to a transcript ID.

        Returns the first transcript ID associated with the given gene ID.

        Args:
            eid: The gene ID (e.g., "ENSMUSG00000000001.4" or "ENSMUSG00000000001").

        Returns:
            The first transcript ID found for that gene ID.

        Raises:
            Polars expression error or IndexError if the gene ID is not found or has no transcripts.
        """
        eid = eid.split(".")[0]
        return self.gtf.filter(pl.col("gene_id") == eid)[0, "transcript_id"]

    def batch_convert(self, val: list[str], src: str, dst: str) -> pl.DataFrame:
        """Batch convert attributes. See available attributes in `self.gtf.columns`.
        Will take the first value found for each attribute.
        !! Will skip non-existent values.

        Args:
            val: list of values to convert.
            src: Attribute to convert from (column name in GTF).
            dst: Attribute to convert to (column name in GTF).

        Returns:
            pl.DataFrame with two columns: `src` and `dst`, containing the mappings.

        Raises:
            ValueError: If none of the input values are found in the `src` column.
        """

        res = pl.DataFrame({src: val}).join(self.gtf.group_by(src).first(), on=src, how="inner")[[src, dst]]
        if not len(res):
            raise ValueError(f"Could not find {val} in {src}")
        if len(res) != len(val):
            logger.warning(
                f"Mapping not bijective. Some values are non-existent in the source column {len(res)} != {len(val)}"
            )
        return res

    def convert(self, val: str, src: str, dst: str) -> str:
        """
        Converts a single value from a source attribute to a destination attribute using GTF data.

        Args:
            val: The value to convert.
            src: The source attribute column name in the GTF data.
            dst: The destination attribute column name in the GTF data.

        Returns:
            The converted value from the `dst` attribute.

        Raises:
            ValueError: If the `val` is not found in the `src` column or if multiple
                matches are found (indicating non-uniqueness).
        """
        res = self.gtf.filter(pl.col(src) == val)[dst]
        if not len(res):
            raise ValueError(f"Could not find {val} in {src}")
        if len(res) > 1:
            raise ValueError(f"Found multiple {val} in {src}")
        return res[0]

    @cache
    def get_transcripts(self, gene: str | None = None, *, eid: str | None = None) -> pl.Series:
        """
        Retrieves transcript IDs for a given gene name or gene ID (Ensembl ID).

        Exactly one of `gene` or `eid` must be provided.

        Args:
            gene: The gene name.
            eid: The gene ID (Ensembl ID).

        Returns:
            A Polars Series containing all transcript IDs associated with the specified gene.
        """
        if gene is not None:
            return self.gtf.filter(pl.col("gene_name") == gene)["transcript_id"]
        return self.gtf.filter(pl.col("gene_id") == eid)["transcript_id"]

    @cache
    def get_seq(self, eid: str, convert: bool = True) -> str:
        """
        Retrieves a sequence from the FASTA file using an ID.

        The ID is typically a transcript ID. If `convert` is True and the ID contains
        a hyphen (suggesting it might be a transcript name), it first attempts to
        convert the transcript name to a transcript ID using `self.convert`.
        The ID (original or converted) is then version-stripped before FASTA lookup.
        If lookup with the version-stripped ID fails, it tries with the original ID.

        Args:
            eid: The identifier (transcript ID or transcript name) for the sequence.
            convert: If True and `eid` contains "-", attempt to convert it from
                'transcript_name' to 'transcript_id' first. Defaults to True.

        Returns:
            The sequence string.

        Raises:
            ValueError: If the ID cannot be found in the FASTA file after attempts,
                or if the sequence is empty.
            ValueError: If `convert` is True and the conversion from transcript name
                to ID fails.
        """
        if "-" in eid and convert:
            eid = self.convert(eid, "transcript_name", "transcript_id")

        try:
            res = self.fa[eid.split(".")[0]].seq
        except KeyError:
            try:
                res = self.fa[eid].seq
            except KeyError:
                raise ValueError(f"Could not find {eid} in fasta file.")

        if not res:
            raise ValueError(f"Could not find {eid}")
        return res

    def filter_gene(self, gene: str) -> pl.DataFrame:
        """
        Filters the GTF DataFrame for entries matching a specific gene name.

        This is a convenience method, equivalent to `self.gtf.filter(pl.col("gene_name") == gene)`.

        Args:
            gene: The gene name to filter by.

        Returns:
            A Polars DataFrame containing only rows related to the specified gene.
        """
        return self.gtf.filter(pl.col("gene_name") == gene)

    @overload
    def __getitem__(self, eid: str) -> pl.Series: ...

    @overload
    def __getitem__(self, eid: list[str]) -> pl.DataFrame: ...

    def __getitem__(self, eid: str | list[str]) -> pl.Series | pl.DataFrame:
        return self.gtf[eid]

    def filter(self, *args: Any, **kwargs: Any):
        return self.gtf.filter(*args, **kwargs)

    @staticmethod
    def parse_gtf(
        path: str | Path | StringIO, filters: Sequence[str] | None = ("transcript",)
    ) -> pl.DataFrame:
        """
        Parses a GTF file into a Polars DataFrame.

        Handles gzipped GTF files. It dynamically discovers attribute keys from the
        GTF's attribute column by sampling rows, then parses these attributes into
        separate columns. 'gene_id' and 'transcript_id' are mandatory and have their
        version suffixes (e.g., .1) stripped.

        Args:
            path: Path to the GTF file or an StringIO object containing GTF data.
            filters: A sequence of feature types (e.g., "transcript", "exon") to keep.
                If None, all features are kept. Defaults to ("transcript",).

        Returns:
            A Polars DataFrame representing the parsed GTF data.

        Raises:
            ValueError: If 'gene_id' or 'transcript_id' attributes are not found in the GTF.
        """
        if not isinstance(path, StringIO) and Path(path).suffix == ".gz":
            path = StringIO(gzip.open(path, "rt").read())
        # fmt: off
        # To get the original keys.
        # list(reduce(lambda x, y: x | json.loads(y), jsoned['jsoned'].to_list(), {}).keys())
        # attr_keys = (
        #     "gene_id", "transcript_id", "gene_type", "gene_name", "gene_biotype", "transcript_type",
        #     "transcript_name", "level", "transcript_support_level", "mgi_id", "tag",
        #     # "havana_gene", "havana_transcript", "protein_id", "ccdsid", "ont",
        # )
        # fmt: on

        df = (
            pl.read_csv(
                path,
                comment_prefix="#",
                separator="\t",
                has_header=False,
                new_columns=[
                    "seqname",
                    "source",
                    "feature",
                    "start",
                    "end",
                    "score",
                    "strand",
                    "frame",
                    "attribute",
                ],
                schema_overrides=[
                    pl.Utf8,
                    pl.Utf8,
                    pl.Utf8,
                    pl.UInt32,
                    pl.UInt32,
                    pl.Utf8,
                    pl.Utf8,
                    pl.Utf8,
                    pl.Utf8,
                ],
            )
            .filter(pl.col("feature").is_in(filters) if filters else pl.col("feature").is_not_null())
            .with_columns(
                pl.concat_str([
                    pl.lit("{"),
                    pl.col("attribute")
                    .str.replace_all(r"; (\w+) ", r', "$1": ')
                    .str.replace_all(";", ",")
                    .str.replace(r"(\w+) ", r'"$1": ')
                    .str.replace(r",$", ""),
                    pl.lit("}"),
                ]).alias("jsoned")
            )
        )

        attr_keys: set[str] = set(
            chain.from_iterable(list(json.loads(s)) for s in df.sample(min(25, len(df)))["jsoned"])
        )
        logger.info(f"Found {len(attr_keys)} attributes in GTF file: {attr_keys}")

        if "gene_id" not in attr_keys:
            raise ValueError("Gene ID not found in GTF file. Required attribute per GTF 2.0 spec.")
        if "transcript_id" not in attr_keys:
            raise ValueError("Transcript ID not found in GTF file. Required attribute per GTF 2.0 spec.")

        df = (
            df.with_columns([
                pl.col("jsoned").str.json_path_match(f"$.{name}").alias(name) for name in attr_keys
            ])
            # .drop(["attribute", "jsoned"])
            .with_columns([
                pl.col("gene_id").str.extract(r"(\w+)(\.\d+)?").alias("gene_id"),
                pl.col("transcript_id").str.extract(r"(\w+)(\.\d+)?").alias("transcript_id"),
            ])
        )
        return pl.DataFrame(df)
