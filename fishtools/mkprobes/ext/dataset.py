import shutil
from functools import cache
from pathlib import Path
from typing import Literal, cast

import polars as pl
from loguru import logger
from pydantic import BaseModel

from fishtools.mkprobes.ext.external_data import ExternalData, ExternalDataDefinition
from fishtools.mkprobes.utils.sequtils import kmers


def parse_jellyfish(path: Path | str) -> pl.DataFrame:
    """Parse a jellyfish output file."""
    try:
        return pl.read_csv(path, separator=" ", has_header=False, new_columns=["kmer", "count"])
    except pl.exceptions.NoDataError:
        raise ValueError(f"Jellyfish file {path} is empty.")


class DatasetDefinition(BaseModel):
    species: str
    external_data: dict[str, ExternalDataDefinition]


class Dataset:
    def __init__(
        self,
        path: str | Path,
        external_data: ExternalData,
        species: str | None = None,
        kmer18_path: str | Path | None = None,
        trna_rna_kmers_path: str | Path | None = None,
    ):
        """

        Args:
            path: _description_
            external_data: _description_
            species: _description_. Defaults to None.
            kmer18_path: _description_. Defaults to None.
            trna_rna_kmers_path: _description_. Defaults to None.
        """
        self.species = species
        self.path = Path(path)
        self.data = external_data
        self.kmer18 = parse_jellyfish(kmer18_path) if kmer18_path else None
        self.kmerset = set(self.kmer18["kmer"] if self.kmer18 is not None else [])
        self.trna_rna_kmers = (
            set(parse_jellyfish(trna_rna_kmers_path)["kmer"]) if trna_rna_kmers_path else None
        )

        # For backwards compatibility
        self.gencode = self.data
        self.ensembl: ExternalData | None = None

    @classmethod
    def from_components(
        cls, path: str | Path, fasta_file: str | Path, *, species: str, overwrite: bool = False
    ):
        path = Path(path)
        fasta_file = Path(fasta_file)

        if path.exists() and not overwrite:
            logger.warning(f"Path {path} already exists. Set overwrite=True to proceed.")

        logger.info(f"Creating dataset at {path}")
        path.mkdir(exist_ok=True, parents=True)
        new_path = path / fasta_file.name
        logger.info(f"Copying {fasta_file} to {new_path}")
        if fasta_file.resolve() != new_path.resolve():
            shutil.copy(fasta_file, new_path)

        external_data = ExternalData(
            cache=path / fasta_file.with_suffix(".parquet").name,
            fasta=fasta_file,
        )
        external_data.bowtie_build()
        external_data.run_jellyfish()

        Path(path / "dataset.json").write_text(
            DatasetDefinition(
                external_data={
                    "default": ExternalDataDefinition(
                        path=path.as_posix(),
                        fasta_name=fasta_file.name,
                        bowtie2_index_name=external_data.bowtie2_index.name,
                        kmer18_name=external_data.kmer.name,
                    )
                },
                species=species,
            ).model_dump_json()
        )

        return cls(path=path, external_data=external_data, species=species, kmer18_path=external_data.kmer)

    @classmethod
    def from_folder(cls, path: Path):
        path = Path(path)
        if not (path / "dataset.json").exists():
            raise FileNotFoundError(f"Path {path} does not exist. Please create a dataset first.")

        definition = DatasetDefinition.model_validate_json((path / "dataset.json").read_text())
        external_data = ExternalData.from_definition(definition.external_data["default"])
        return cls(
            path=path,
            external_data=external_data,
            species=definition.species,
            kmer18_path=path / external_data.kmer,
        )

    def check_kmers(self, seq: str):
        if not self.trna_rna_kmers:
            logger.warning("No tRNA-RNA kmers found. Skipping.")
            return False

        return any(x in self.trna_rna_kmers for x in kmers(seq, 18))

    @property
    @cache
    def appris(self):
        raise NotImplementedError("appris not implemented")


class ReferenceDataset(Dataset):
    def __init__(self, path: Path | str):
        self.path = path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        super().__init__(
            path,
            external_data=ExternalData(
                cache=self.path / "gencode.parquet",
                gtf_path=self.path / "gencode.gtf.gz",
                fasta=self.path / "cdna_ncrna_trna.fasta",
            ),
            kmer18_path=self.path / "kmer18.jf",
            trna_rna_kmers_path=self.path / "r_t_snorna15.jf",
            species=cast(Literal["human", "mouse"], self.path.name),
        )
        if self.species not in ("human", "mouse"):
            raise ValueError(f"Species not human or mouse, got {self.species}")

        try:
            self.ensembl = ExternalData(
                cache=self.path / "ensembl.parquet",
                gtf_path=self.path / "ensembl.gtf.gz",
                fasta=self.path / "cdna_ncrna_trna.fasta",
            )
        except FileNotFoundError:
            if self.species in ("human", "mouse"):
                raise FileNotFoundError(f"No GENCODE data found for {self.species}.")
            self.ensembl = None

    @property
    @cache
    def appris(self):
        try:
            return pl.read_csv(
                self.path / "appris_data.principal.txt",
                separator="\t",
                has_header=False,
                new_columns=["gene_name", "gene_id", "transcript_id", "ccds", "annotation"],
            )
        except FileNotFoundError:
            logger.warning("No APPRIS data found.")
            return None
