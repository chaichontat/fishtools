# %%
import gzip
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Sequence, TypedDict, cast, overload

import mygene
import polars as pl
import pyfastx
from Bio import SeqIO
from oligocheck.boilerplate import jprint
from oligocheck.merfish.alignment import gen_fasta

mg = mygene.MyGeneInfo()


class ExternalData:
    def __init__(self, cache: Path | str, *, path: Path | str, fasta: Path | str) -> None:
        self.fa = pyfastx.Fasta(fasta, key_func=lambda x: x.split(" ")[0].split(".")[0])
        self._ts_gene_map: dict[str, str] | None = None

        if Path(cache).exists():
            self.gtf: pl.DataFrame = pl.read_parquet(cache)
        else:
            self.gtf = self.parse_gtf(path)
            self.gtf.write_parquet(cache)

    @cache
    def gene_info(self, gene: str) -> pl.DataFrame:
        return self.gtf.filter(pl.col("gene_name") == gene)

    @cache
    def gene_to_eid(self, gene: str) -> str:
        try:
            return self.gene_info(gene)[0, "gene_id"]
        except pl.ComputeError:
            raise ValueError(f"Could not find {gene}")

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
        except pl.ComputeError:
            return eid

    @cache
    def eid_to_ts(self, eid: str) -> str:
        eid = eid.split(".")[0]
        return self.gtf.filter(pl.col("gene_id") == eid)[0, "transcript_id"]

    @cache
    def get_transcripts(self, gene: str | None = None, *, eid: str | None = None) -> pl.Series:
        if gene is not None:
            return self.gtf.filter(pl.col("gene_name") == gene)["transcript_id"]
        return self.gtf.filter(pl.col("gene_id") == eid)["transcript_id"]

    @cache
    def get_seq(self, eid: str) -> str:
        res = self.fa[eid.split(".")[0]].seq
        if not res:
            raise ValueError(f"Could not find {eid}")
        return res

    def filter_gene(self, gene: str) -> pl.DataFrame:
        return self.gtf.filter(pl.col("gene_name") == gene)

    @overload
    def __getitem__(self, eid: str) -> pl.Series:
        ...

    @overload
    def __getitem__(self, eid: list[str]) -> pl.DataFrame:
        ...

    def __getitem__(self, eid: str | list[str]):
        return self.gtf[eid]

    def filter(self, *args: Any, **kwargs: Any):
        return self.gtf.filter(*args, **kwargs)

    @staticmethod
    def parse_gtf(path: str | Path, filters: Sequence[str] = ("transcript",)) -> pl.DataFrame:
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
            pl.scan_csv(
                path,
                comment_char="#",
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
                dtypes=[pl.Utf8, pl.Utf8, pl.Utf8, pl.UInt32, pl.UInt32, pl.Utf8, pl.Utf8, pl.Utf8, pl.Utf8],
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
            .collect()
        )

    def check_gene_names(self, genes: list[str]):
        notfound = []
        ok: list[str] = []
        for gene in genes:
            try:
                self.gene_to_eid(gene)
                ok.append(gene)
            except ValueError:
                print(f"Gene {gene} not found in gtf")
                notfound.append(gene)
        converted, res = find_aliases(notfound)

        return (
            ok + [x["symbol"] for x in converted.values()],
            {k: v["symbol"] for k, v in converted.items()},
            res,
        )


# %%
def get_rrna(path: str) -> set[str]:
    # Get tRNA and rRNA
    out = []
    ncrna = SeqIO.parse(gzip.open(path, "rt") if path.endswith(".gz") else path, "fasta")
    # Example line
    # ENSMUST00000104605.4 ncrna chromosome:GRCm39:Y:1308273:1308377:-1 gene:ENSMUSG00000077793.4 gene_biotype:snRNA transcript_biotype:snRNA gene_symbol:Gm25565 description:predicted gene, 25565 [Source:MGI Symbol;Acc:MGI:5455342]
    for line in ncrna:
        attrs = line.description.split(", ")[0].split(" ")
        attrs[0] = "transcript_id:" + attrs[0]
        attrs[1] = "type:" + attrs[1]
        actual = attrs[:7]
        description = " ".join(attrs[7:]).split(":", maxsplit=1)
        attrs = dict(
            [
                ["seq", str(line.seq)],
                ["description", description[1].strip() if len(description) > 1 else ""],
                *[x.split(":", maxsplit=1) for x in actual],
            ]
        )
        out.append(attrs)

    return set(pl.from_records(out).filter(pl.col("gene_biotype") == "rRNA")["transcript_id"])
    # with open(path, "w") as f:
    #     for _, row in out[out.gene_biotype == "rRNA"].iterrows():
    #         f.write(f">{row.transcript_id}\n{row.seq}\n")

    # return set(pd.read_table(path, header=None)[0].str.split(">", expand=True)[1].dropna())


def gen_rrna_trna(gtf: ExternalData):
    rrna = gtf.gtf.filter(pl.col("gene_biotype") == "rRNA")
    if not len(rrna):
        raise ValueError("No rRNA found in GTF. Use Ensembl GTF.")
    seqs = [gtf.get_seq(x) for x in rrna["transcript_id"]]
    Path("data/mm39/rrna.fasta").write_text(gen_fasta(rrna["transcript_id"], seqs).getvalue())


# %%