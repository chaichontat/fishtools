# %%
import gzip
import shlex
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Collection, Literal, NamedTuple, TypedDict

import polars as pl
import pyfastx
from loguru import logger as log

from fishtools.utils.io import download, get_file_name, set_cwd
from fishtools.utils.utils import check_if_exists, check_if_posix, run_process

from ..utils._alignment import gen_fasta
from .external_data import _ExternalData

url_files = {
    "mouse": Path(__file__).parent / "mouseurls.tsv",
    "human": Path(__file__).parent / "humanurls.tsv",
}

Gtfs = NamedTuple("gtfs", [("ensembl", _ExternalData), ("gencode", _ExternalData)])


class NecessaryFiles(TypedDict):
    cdna: str
    ensembl_gtf: str
    gencode_gtf: str
    ncrna: str
    trna: str
    appris: str


def _process_tsv(file: Path | str) -> NecessaryFiles:
    file = Path(file)
    return NecessaryFiles([x.split("\t") for x in file.read_text().splitlines()])  # type: ignore


@check_if_posix
@check_if_exists(log, lambda kwargs: kwargs["out"])
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
    out = Path(out).resolve()
    if isinstance(seqs, str):
        stdin = Path(seqs).read_bytes()
    else:
        stdin = gen_fasta(seqs).getvalue().encode()

    stdout1 = run_process(
        shlex.split(
            rf"jellyfish count -o /dev/stdout -m {kmer} -t {thread} -s {hash_size} -L {minimum} -c {counter} {'--both-strands' if both_strands else ''} /dev/stdin"
        ),
        stdin,
    )
    stdout2 = run_process(shlex.split("jellyfish dump -c -L 1 /dev/stdin"), stdout1)

    Path(out).write_bytes(stdout2)


def download_gtf_fasta(path: Path | str, species: Literal["mouse", "human"]):
    (path := Path(path)).mkdir(exist_ok=True, parents=True)
    urls = _process_tsv(uf := url_files[species])
    shutil.copy(uf, path / "urls.tsv")
    del uf

    filenames = {k: get_file_name(v) for k, v in urls.items()}  # type: ignore

    with set_cwd(path):
        todownload = {k: v for k, v in urls.items() if not Path(v).exists()}  # type: ignore
        with ThreadPoolExecutor() as pool:
            names = [k[:-4] + ".gtf.gz" if "gtf" in k else filenames[k] for k in todownload]
            pool.map(download, todownload.values(), ["."] * len(urls), names)

        for ori, v in zip(todownload.values(), names):
            print(ori, v)
            if not Path(v or ori).exists():
                raise FileNotFoundError(f"Could not download {v or ori}")

        if filenames["trna"].endswith(".gz"):
            trna_name = filenames["trna"].split(".")[0] + ".fa"
            with tarfile.open(filenames["trna"], "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name == trna_name:
                        tar.extract(member, ".")
                        break
            filenames["trna"] = trna_name

        # Concatenate fasta and tRNA files
        if (p_ := Path("cdna_ncrna_trna.fasta")).exists():
            log.info("cdna_ncrna_trna.fasta already exists. Skipping concatenation.")
        else:
            log.info("Concatenating fasta and tRNA files")
            out = b""
            for filename in [filenames["cdna"], filenames["ncrna"]]:
                with gzip.open(filename, "rb") as input_file:
                    out += input_file.read()
                    out += b"\n"
                with open(filenames["trna"], "rb") as f:
                    out += f.read()
                    out += b"\n"
            p_.write_bytes(out)


def get_rrna_snorna(gtf: _ExternalData):
    ids = gtf.gtf.filter(pl.col("gene_biotype").is_in(["rRNA", "snoRNA"]))["transcript_id"]
    return ids, list(map(gtf.get_seq, ids))


@check_if_exists(log, lambda kwargs: Path(kwargs["path"]) / "cdna18.jf")
def run_jellyfish(path: Path | str):
    path = Path(path)
    with set_cwd(path):
        urls = _process_tsv("urls.tsv")
        filenames = {k: get_file_name(v) for k, v in urls.items()}  # type: ignore

        gtf = _ExternalData("ensembl.parquet", fasta="cdna_ncrna_trna.fasta", gtf_path="ensembl.gtf.gz")
        log.info("Getting tRNA, rRNA, and snoRNA sequences.")

        toexclude = [x.seq for x in pyfastx.Fasta(filenames["trna"].split(".")[0] + ".fa")] + get_rrna_snorna(
            gtf
        )[1]

        log.info("Running jellyfish for 15-mers in r, t, and, snoRNAs.")
        jellyfish(toexclude, "r_t_snorna15.jf", 15)

        log.info("Running jellyfish for 18-mers in cDNA.")
        jellyfish(
            [x.seq for x in pyfastx.Fasta("cdna_ncrna_trna.fasta")],
            "cdna18.jf",
            18,
            minimum=10,
            counter=4,
        )
        if not Path("cdna18.jf").exists():
            raise FileNotFoundError("cdna18.jf not found. Jellyfish run failed.")


@check_if_exists(
    log, lambda kwargs: Path(kwargs["fasta"]).with_suffix(Path(kwargs["fasta"]).suffix + ".masked")
)
def run_repeatmasker(fasta: Path | str, species: str, threads: int = 16):
    fasta = Path(fasta)
    subprocess.run(
        shlex.split(f'RepeatMasker -norna -pa {threads} -norna -species "{species}" {fasta.as_posix()}'),
        check=True,
    )


# %%
if __name__ == "__main__":
    gtf = _ExternalData("data/mouse/ensembl.parquet", fasta="data/mouse/cdna_ncrna_trna.fasta")
# get("data/mouse")

# GTF https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.annotation.gtf.gz
# Ensembl https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.toplevel.fa.gz

# Run jellyfish on tRNA, rRNA, and snoRNA for 18-mer exclusion.
# http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-tRNAs.tar.gz

# %%
