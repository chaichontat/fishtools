import io
import shlex
import subprocess
from pathlib import Path
from typing import Any, Collection, Iterable

from Bio import AlignIO
from loguru import logger

from fishtools.utils.utils import check_if_exists, check_if_posix, run_process


def _n_generator() -> Iterable[str]:
    i = 0
    while True:
        yield str(i)
        i += 1


def gen_fastq(seqs: Iterable[str], *, names: Iterable[Any] | None = None) -> io.StringIO:
    f = io.StringIO()
    if names is None:
        names = _n_generator()
    for name, seq in zip(map(str, names), seqs):
        f.write(f"@{name}\n")
        f.write(seq + "\n")
        f.write("+\n")
        f.write("~" * len(seq) + "\n")
    return f


def gen_fasta(seqs: Iterable[str], *, names: Iterable[Any] | None = None) -> io.StringIO:
    f = io.StringIO()
    if names is None:
        names = _n_generator()
    for name, seq in zip(map(str, names), seqs):
        f.write(f">{name}\n")
        f.write(seq + "\n")
    return f


@check_if_exists(logger, lambda kwargs: f"{Path(kwargs['fasta_path']).parent}/{kwargs['name']}.1.bt2")
def bowtie_build(fasta_path: Path | str, name: str) -> bytes | None:
    fasta_path = Path(fasta_path).resolve()
    logger.info(f"Generating bowtie2 index for {name}")
    res = subprocess.run(
        ["bowtie2-build", fasta_path.as_posix(), (fasta_path.parent / name).as_posix()],
        check=True,
    ).stdout
    if not Path(fasta_path.parent / f"{name}.1.bt2").exists():
        raise FileNotFoundError(f"Bowtie2 index not found for {name}. Bowtie2 build failed.")
    return res


def gen_bowtie_index(fasta: str, path: Path | str, name: str) -> bytes | None:
    logger.info(f"Generating bowtie2 index for {name}")
    (path := Path(path)).mkdir(exist_ok=True, parents=True)
    (path / (name + ".fasta")).write_text(fasta)
    return bowtie_build(path / (name + ".fasta"), name)


def run_bowtie(
    stdin: str | bytes,
    reference: str | Path,
    capture_stderr: bool = False,
    seed_length: int = 15,
    n_return: int = 100,
    threads: int = 16,
    threshold: int = 15,
    fasta: bool = False,
    no_reverse: bool = False,
    no_forward: bool = False,
) -> str:
    logger.info(f"Running bowtie2 with {reference}")

    try:
        res = subprocess.run(
            shlex.split(
                # A base that matches receives a bonus of +2 be default.
                # A mismatched base at a high-quality position in the read receives a penalty of -6 by default.
                # --no-hd No SAM header
                # -k 100 report up to 100 alignments per read
                # -D 20 consecutive seed extension attempts can "fail" before Bowtie 2 moves on
                # -R 3 the maximum number of times Bowtie 2 will "re-seed" reads with repetitive seeds.
                # -L 17 seed length
                # -i C,2 Seed interval, every 2 bp
                # --score-min G,1,4 f(x) = 1 + 4*ln(read_length)
                # --score-min L,0,-0.6 f(x) = -0.6*read_length
                f"bowtie2 -x {reference} -U - "
                f"--no-hd -t {f'-k {n_return}' if n_return > 0 else '-a'} --local -D 20 -R 3 "
                f"--score-min L,{threshold * 2},0 --mp 1,1 --ignore-quals {'-f ' if fasta else ''}"
                f"-N 0 -L {seed_length} -i C,2 -p {threads} {'--norc ' if no_reverse else ''} {'--nofw ' if no_forward else ''}"
            ),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else None,
            encoding="ascii" if isinstance(stdin, str) else None,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e
    return res


def run_mafft(stdin: str | bytes) -> str:
    return subprocess.run(
        ["mafft-linsi", "--op", "2", "-"],
        input=stdin,
        encoding="ascii" if isinstance(stdin, str) else None,
        capture_output=True,
        check=True,
    ).stdout


def parse_mafft(s: str) -> AlignIO.MultipleSeqAlignment:
    return AlignIO.read(io.StringIO(s), "fasta")


@check_if_posix
@check_if_exists(logger, lambda kwargs: kwargs["out"])
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
