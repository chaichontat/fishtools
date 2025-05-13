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
    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"ExternalData not created with GTF path. Cannot access {name}")

    def __getitem__(self, name: str | list[str]) -> Any:
        raise NotImplementedError("ExternalData not created with GTF path. Cannot index.")


class ExternalDataDefinition(BaseModel):
    path: str
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
        - The default function (`lambda x: x.split(" ")[0].split(".")[0]`) attempts
          to get the ID before the first space and remove any version suffix (like .1).
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

    def __init__(
        self,
        cache: Path | str | None = None,
        *,
        fasta: Path | str,
        gtf_path: Path | str | None = None,
        regen_cache: bool = False,
        fasta_key_func: Callable[[str], str | None] = lambda x: x.split(" ")[0].split(".")[0],
        bowtie2_index: str | None = None,
        kmer18: str | None = None,
    ) -> None:
        self.fasta_path = Path(fasta)
        self.fa = pyfastx.Fasta(Path(fasta).as_posix(), key_func=fasta_key_func)
        self._ts_gene_map: dict[str, str] | None = None

        if cache and Path(cache).exists() and not regen_cache:
            self.gtf: pl.DataFrame | MockGTF = pl.read_parquet(cache)
        else:
            if gtf_path is None:
                logger.warning("GTF path not specified. Must be specified for reference species.")
                self.gtf = MockGTF()
            else:
                if not cache:
                    raise ValueError("Cache path must be specified if GTF path is provided.")
                self.gtf = self.parse_gtf(Path(gtf_path).resolve())
                self.gtf.write_parquet(cache)

        self._override_bowtie2_index = bowtie2_index
        self._override_kmer18 = kmer18

    @classmethod
    def from_definition(cls, definition: ExternalDataDefinition):
        path = Path(definition.path)
        return cls(
            cache=path / definition.cache_name if definition.cache_name else None,
            fasta=path / definition.fasta_name,
            gtf_path=path / definition.gtf_name if definition.gtf_name else None,
            bowtie2_index=definition.bowtie2_index_name,
            kmer18=definition.kmer18_name,
        )

    @property
    def bowtie2_index(self):
        if self._override_bowtie2_index:
            return self.fasta_path.parent / self._override_bowtie2_index

        bt = self.fasta_path.parent.glob(f"{self.fasta_path.stem}*.bt2")
        if not len(list(bt)):
            raise FileNotFoundError(
                f"Bowtie2 index not found for {self.fasta_path.stem}. Please build with `bowtie2-build {{fasta_path}} {{fasta file name}}` or call self.bowtie_build()."
            )

        return self.fasta_path.with_suffix("")

    def bowtie_build(self):
        return bowtie_build(self.fasta_path, self.fasta_path.stem)

    @property
    def kmer(self):
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
        if self.fasta_path.with_suffix(".jf").exists() and not overwrite:
            logger.info(f"Jellyfish file {self.fasta_path.with_suffix('.jf')} already exists. Skipping.")
            return

        logger.info("Running jellyfish for 18-mers in cDNA.")
        jellyfish(
            [x.seq for x in self.fa],
            self.fasta_path.with_suffix(".jf"),
            kmer,
            minimum=10,
            counter=4,
        )
        if not self.fasta_path.with_suffix(".jf").exists():
            raise FileNotFoundError("cdna18.jf not found. Jellyfish run failed.")

    @cache
    def gene_info(self, gene: str) -> pl.DataFrame:
        return self.gtf.filter(pl.col("gene_name") == gene)

    @cache
    def gene_to_eid(self, gene: str) -> pl.Series:
        ret = self.gene_info(gene)
        if ret.is_empty():
            raise ValueError(f"Could not find {gene}")
        return ret[:, "gene_id"]

    @cache
    def ts_to_gene(self, ts: str) -> str:
        ts = ts.split(".")[0]
        if self._ts_gene_map is None:
            self._ts_gene_map = {k: v for k, v in zip(self.gtf["transcript_id"], self.gtf["gene_name"])}
        return self._ts_gene_map.get(ts, ts)

    @cache
    def ts_to_tsname(self, eid: str) -> str | None:
        eid = eid.split(".")[0]
        try:
            return self.gtf.filter(pl.col("transcript_id") == eid)["transcript_name"].first()  # type: ignore
        except pl.exceptions.ComputeError:
            return eid

    @cache
    def eid_to_ts(self, eid: str) -> str:
        eid = eid.split(".")[0]
        return self.gtf.filter(pl.col("gene_id") == eid)[0, "transcript_id"]

    def batch_convert(self, val: list[str], src: str, dst: str) -> pl.DataFrame:
        """Batch convert attributes. See available attributes in `self.gtf.columns`.
        Will take the first value found for each attribute.
        !! Will skip non-existent values.

        Args:
            val: list of values to convert.
            src: Attribute to convert from.
            dst: Attribute to convert to.

        Returns:
            pl.DataFrame[[src, dst]]
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
        res = self.gtf.filter(pl.col(src) == val)[dst]
        if not len(res):
            raise ValueError(f"Could not find {val} in {src}")
        if len(res) > 1:
            raise ValueError(f"Found multiple {val} in {src}")
        return res[0]

    @cache
    def get_transcripts(self, gene: str | None = None, *, eid: str | None = None) -> pl.Series:
        if gene is not None:
            return self.gtf.filter(pl.col("gene_name") == gene)["transcript_id"]
        return self.gtf.filter(pl.col("gene_id") == eid)["transcript_id"]

    @cache
    def get_seq(self, eid: str, convert: bool = True) -> str:
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
