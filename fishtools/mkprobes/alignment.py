import io
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from Bio import AlignIO


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


def gen_bowtie_index(fasta: str, path: str, name: str) -> bytes:
    Path(path).mkdir(exist_ok=True, parents=True)
    (Path(path) / (name + ".fasta")).write_text(fasta)
    return subprocess.run(
        shlex.split(f"bowtie2-build {path}/{name}.fasta {path}/{name}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def run_bowtie(
    stdin: str | bytes,
    reference: str,
    capture_stderr: bool = False,
    seed_length: int = 15,
    n_return: int = 100,
    threads: int = 32,
    threshold: int = 15,
    fasta: bool = False,
) -> str:
    return subprocess.run(
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
            f"--score-min L,{threshold*2},0 --mp 1,1 --ignore-quals {'-f ' if fasta else ''}"
            f"-N 0 -L {seed_length} -i C,2 -p {threads}"
        ),
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else None,
        encoding="ascii" if isinstance(stdin, str) else None,
        check=True,
    ).stdout


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
