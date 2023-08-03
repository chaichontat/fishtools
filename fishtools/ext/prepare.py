# %%
import gzip
import mmap
import shlex
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path
from typing import Collection, Literal, NamedTuple, TypedDict

import polars as pl
import pyfastx
from loguru import logger as log

from fishtools.ext.external_data import ExternalData
from fishtools.io.io import download, get_file_name, set_cwd
from fishtools.mkprobes.initial_screen.alignment import gen_fasta
from fishtools.utils import check_if_posix

url_files = {
    "mouse": Path(__file__).parent / "../../" / "static" / "mouseurls.tsv",
    "human": Path(__file__).parent / "../../" / "static" / "humanurls.tsv",
}

Gtfs = NamedTuple("gtfs", [("ensembl", ExternalData), ("gencode", ExternalData)])


class NecessaryFiles(TypedDict):
    cdna: str
    ensembl_gtf: str
    gencode_gtf: str
    ncrna: str
    trna: str


def _process_tsv(file: Path | str) -> NecessaryFiles:
    file = Path(file)
    return NecessaryFiles([x.split("\t") for x in file.read_text().splitlines()])  # type: ignore


@check_if_posix
def jellyfish(
    seqs: Collection[str] | str,
    out: str | Path,
    kmer: int,
    *,
    hash_size: str = "10G",
    minimum: int = 1,
    counter: int = 2,
    thread: int = 16,
    both_strands: bool = False,
):
    """
    http://www.cs.cmu.edu/~ckingsf/software/jellyfish/jellyfish-manual-1.1.pdf
    """
    if isinstance(seqs, str):
        stdin = Path(seqs).read_text()
    else:
        stdin = gen_fasta(map(str, range(len(seqs))), seqs).getvalue()

    res = subprocess.run(
        shlex.split(
            rf"jellyfish count -o /dev/stdout -m {kmer} -t {thread} -s {hash_size} -L {minimum} -c {counter} {'--both-strands' if both_strands else ''} /dev/stdin"
        ),
        input=stdin,
        stdout=subprocess.PIPE,
        check=True,
    )
    res = subprocess.run(
        # -c == column format
        shlex.split(rf"jellyfish dump -c -L 1 /dev/stdin > {out}"),
        input=res.stdout,
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )


def download_gtf_fasta(path: Path | str, species: Literal["mouse", "human"]) -> Gtfs:
    path = Path(path)
    urls = _process_tsv(uf := url_files[species])
    shutil.copy(uf, path / "urls.tsv")
    del uf

    filenames = {k: get_file_name(v) for k, v in urls.items()}  # type: ignore

    with set_cwd(path):
        todownload = {k: v for k, v in urls.items() if not Path(v).exists()}  # type: ignore
        with ThreadPoolExecutor() as pool:
            names = [k[:-4] + ".gtf.gz" if "gtf" in k else None for k in todownload]
            pool.map(download, todownload.values(), ["."] * len(urls), names)

        trna_name = filenames["trna"].split(".")[0] + ".fa"
        with tarfile.open(filenames["trna"], "r:gz") as tar:
            for member in tar.getmembers():
                if member.name == trna_name:
                    tar.extract(member, ".")
                    break
        filenames["trna"] = trna_name

        # Concatenate fasta and tRNA files
        log.info("Concatenating fasta and tRNA files")
        out = b""

        for filename in [filenames["cdna"], filenames["ncrna"]]:
            with gzip.open(filename, "rb") as input_file:
                out += input_file.read()
                out += b"\n"
            with open(filenames["trna"], "rb") as f:
                out += f.read()
                out += b"\n"
        Path("cdna_ncrna_trna.fasta").write_bytes(out)

        return Gtfs(
            ensembl=ExternalData(
                cache="ensembl.parquet",
                gtf_path="ensembl.gtf.gz",
                fasta="cdna_ncrna_trna.fasta",
                regen_cache=True,
            ),
            gencode=ExternalData(
                cache="gencode.parquet",
                gtf_path="gencode.gtf.gz",
                fasta="cdna_ncrna_trna.fasta",
                regen_cache=True,
            ),
        )


def get_rrna_snorna(gtf: ExternalData):
    ids = gtf.gtf.filter(pl.col("gene_biotype").is_in(["rRNA", "snoRNA"]))["transcript_id"]
    return ids, list(map(gtf.get_seq, ids))


def run_jellyfish(path: Path | str):
    path = Path(path)
    with set_cwd(path):
        urls = _process_tsv("urls.tsv")
        filenames = {k: get_file_name(v) for k, v in urls.items()}  # type: ignore

        gtf = ExternalData("ensembl.parquet", fasta="cdna_ncrna_trna.fasta")
        log.info("Getting tRNA, rRNA, and snoRNA sequences.")

        toexclude = [x.seq for x in pyfastx.Fasta(filenames["trna"].split(".")[0] + ".fa")] + get_rrna_snorna(
            gtf
        )[1]

        log.info("Running jellyfish for 15-mers in r, t, snoRNAs.")
        jellyfish(toexclude, path / "rtsno.jf", 15)

        log.info("Running jellyfish for 18-mers in cDNA.")
        jellyfish(
            [x.seq for x in pyfastx.Fasta("cdna_ncrna_trna.fasta")],
            path / "cdna.jf",
            18,
            minimum=10,
            counter=4,
        )


# %%
if __name__ == "__main__":
    gtf = ExternalData("data/mouse/ensembl.parquet", fasta="data/mouse/cdna_ncrna_trna.fasta")
# get("data/mouse")

# GTF https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.annotation.gtf.gz
# Ensembl https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.toplevel.fa.gz

# Run jellyfish on tRNA, rRNA, and snoRNA for 18-mer exclusion.
# http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-tRNAs.tar.gz

# %%
