# %%
import gzip
import json
from functools import cache
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Sequence, cast, overload

import mygene
import polars as pl
import pyfastx
import requests
from loguru import logger as log

from fishtools.mkprobes.utils.sequtils import kmers

mg = mygene.MyGeneInfo()


def get_ensembl(path: Path | str, id_: str):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    if (p := (path / f"{id_}.json")).exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning(f"Error decoding {p}. Deleting.")
            p.unlink(missing_ok=True)

    log.info(f"Fetching {id_} on ensembl")
    res = requests.get(f"https://rest.ensembl.org/lookup/id/{id_}?content-type=application/json", timeout=30)
    res.raise_for_status()
    p.write_text(json.dumps(j := res.json(), indent=2))
    return j


class _ExternalData:
    def __init__(
        self,
        cache: Path | str,
        *,
        fasta: Path | str,
        gtf_path: Path | str | None = None,
        regen_cache: bool = False,
    ) -> None:
        self.fa = pyfastx.Fasta(Path(fasta).as_posix(), key_func=lambda x: x.split(" ")[0].split(".")[0])
        self._ts_gene_map: dict[str, str] | None = None

        if Path(cache).exists() and not regen_cache:
            self.gtf: pl.DataFrame = pl.read_parquet(cache)
        else:
            if gtf_path is None:
                raise ValueError("gtf_path must be specified if cache does not exist.")
            log.info("Parsing external GTF")
            self.gtf = self.parse_gtf(Path(gtf_path).resolve())
            self.gtf.write_parquet(cache)

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
    def ts_to_tsname(self, eid: str) -> str:
        eid = eid.split(".")[0]
        try:
            return self.gtf.filter(pl.col("transcript_id") == eid)[0, "transcript_name"]
        except pl.exceptions.ComputeError:
            return eid

    @cache
    def eid_to_ts(self, eid: str) -> str:
        eid = eid.split(".")[0]
        return self.gtf.filter(pl.col("gene_id") == eid)[0, "transcript_id"]

    def batch_convert(self, val: list[str], src: str, dst: str) -> pl.DataFrame:
        """Batch convert attributes. See available attributes in `self.gtf.columns`.
        Will take the first value found for each attribute.

        Args:
            val: list of values to convert.
            src: Attribute to convert from.
            dst: Attribute to convert to.

        Returns:
            pl.DataFrame[[src, dst]]
        """
        res = pl.DataFrame({src: val}).join(self.gtf.group_by(src).first(), on=src, how="left")[[src, dst]]
        if not len(res):
            raise ValueError(f"Could not find {val} in {src}")
        if len(res) != len(val):
            log.warning(f"Mapping not bijective. {len(res)} != {len(val)}")
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
    def get_seq(self, eid: str) -> str:
        if "-" in eid:
            eid = self.convert(eid, "transcript_name", "transcript_id")

        res = self.fa[eid.split(".")[0]].seq
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
    def parse_gtf(path: str | Path | StringIO, filters: Sequence[str] = ("transcript",)) -> pl.DataFrame:
        if not isinstance(path, StringIO) and Path(path).suffix == ".gz":
            path = StringIO(gzip.open(path, "rt").read())
        # fmt: off
        # To get the original keys.
        # list(reduce(lambda x, y: x | json.loads(y), jsoned['jsoned'].to_list(), {}).keys())
        attr_keys = (
            "gene_id", "transcript_id", "gene_type", "gene_name", "gene_biotype", "transcript_type",
            "transcript_name", "level", "transcript_support_level", "mgi_id", "tag",
            # "havana_gene", "havana_transcript", "protein_id", "ccdsid", "ont",
        )
        # fmt: on

        return (
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
                pl.concat_str(
                    [
                        pl.lit("{"),
                        pl.col("attribute")
                        .str.replace_all(r"; (\w+) ", r', "$1": ')
                        .str.replace_all(";", ",")
                        .str.replace(r"(\w+) ", r'"$1": ')
                        .str.replace(r",$", ""),
                        pl.lit("}"),
                    ]
                ).alias("jsoned")
            )
            .with_columns(
                [
                    pl.col("jsoned").str.json_path_match(f"$.{name}").alias(name)
                    # .cast(pl.Categorical if "type" in name or "tag" == name else pl.Utf8)
                    for name in attr_keys
                ]
            )
            # .drop(["attribute", "jsoned"])
            .with_columns(
                [
                    pl.col("gene_id").str.extract(r"(\w+)(\.\d+)?").alias("gene_id"),
                    pl.col("transcript_id").str.extract(r"(\w+)(\.\d+)?").alias("transcript_id"),
                ]
            )
        )


# @dataclass(frozen=True)
class Dataset:
    def __init__(self, path: Path | str, weird_organisms: bool = False):
        self.path = Path(path)
        self.species = cast(Literal["human", "mouse"], self.path.name)
        if self.species not in ("human", "mouse"):
            log.warning(f"Species not human or mouse, got {self.species}")

        self.gencode = _ExternalData(
            cache=self.path / "gencode.parquet",
            gtf_path=self.path / "gencode.gtf.gz",
            fasta=self.path / "cdna_ncrna_trna.fasta",
        )

        self.ensembl = _ExternalData(
            cache=self.path / "ensembl.parquet",
            gtf_path=self.path / "ensembl.gtf.gz",
            fasta=self.path / "cdna_ncrna_trna.fasta",
        )

        self.kmer18 = pl.read_csv(
            self.path / "cdna18.jf", separator=" ", has_header=False, new_columns=["kmer", "count"]
        )
        self.trna_rna_kmers = set(
            pl.read_csv(
                self.path / "r_t_snorna15.jf", separator=" ", has_header=False, new_columns=["kmer", "count"]
            )["kmer"]
        )
        self.kmerset = set(self.kmer18["kmer"])

    @property
    @cache
    def appris(self):
        return pl.read_csv(
            self.path / "appris_data.principal.txt",
            separator="\t",
            has_header=False,
            new_columns=["gene_name", "gene_id", "transcript_id", "ccds", "annotation"],
        )

    def check_kmers(self, seq: str):
        # fmt: off
        return (
            # any(x in self.kmerset for x in kmers(seq, 18))
            any(x in self.trna_rna_kmers for x in kmers(seq, 18))
        )
        # fmt: on

    # def check_gene_names(self, genes: list[str]):
    #     notfound = []
    #     ok: list[str] = []
    #     for gene in genes:
    #         try:
    #             self.gene_to_eid(gene)
    #             ok.append(gene)
    #         except ValueError:
    #             print(f"Gene {gene} not found in gtf")
    #             notfound.append(gene)
    #     converted, res = find_aliases(notfound)

    #     return (
    #         ok + [x["symbol"] for x in converted.values()],
    #         {k: v["symbol"] for k, v in converted.items()},
    #         res,
    #     )
