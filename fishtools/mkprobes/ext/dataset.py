import shutil
from functools import cache
from pathlib import Path
from typing import Literal, cast

import click
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
        raise ValueError(f"Jellyfish file {path} is empty. Please delete and rerun create-dataset.")


class DatasetDefinition(BaseModel):
    """
    Defines the structure for storing dataset configuration.

    This Pydantic model is used to serialize and deserialize dataset metadata,
    primarily for saving and loading dataset configurations from a JSON file.
    It ensures that the dataset information is consistent and correctly structured.

    - Ensure all required fields (species, external_data) are present when creating an instance.
    """

    species: str
    external_data: dict[str, ExternalDataDefinition]


class Dataset:
    """
    Represents a dataset for probe design, including sequence data and k-mer information.

    This class manages the core data files (FASTA, k-mer counts) and provides
    methods to access and process them. It can be initialized either by building
    from component files (FASTA) or by loading an existing dataset from a folder.

    Intended Use:
        - Creating new datasets from FASTA files.
        - Loading pre-existing datasets for analysis or probe design.
        - Accessing k-mer sets and performing k-mer-based checks.

    Potential Pitfalls:
        - If `from_components` is not used or if files are manually moved/deleted,
          the dataset might become inconsistent, leading to `FileNotFoundError` or
          `ValueError` during operations.
        - Ensure `kmer18_path` and `trna_rna_kmers_path` point to valid Jellyfish output
          files if provided; otherwise, related functionalities might not work as expected.
        - The `overwrite` flag in `from_components` should be used with caution to avoid
          accidental data loss.
    """

    def __init__(
        self,
        path: str | Path,
        external_data: ExternalData,
        species: str | None = None,
        kmer18_path: str | Path | None = None,
        trna_rna_kmers_path: str | Path | None = None,
    ):
        """
        Initializes a Dataset object.

        Args:
            path: The base directory path for the dataset.
            external_data: An `ExternalData` object containing sequence and annotation data.
            species: The species name (e.g., "human", "mouse"). Defaults to None.
            kmer18_path: Path to the 18-mer Jellyfish count file. Defaults to None.
            trna_rna_kmers_path: Path to a Jellyfish count file for tRNA/rRNA kmers.
                Defaults to None.
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
        cls,
        path: str | Path,
        fasta_file: str | Path,
        *,
        species: str,
        overwrite: bool = False,
    ):
        """
        Creates a new dataset from a FASTA file.

        This method will:
        1. Create the dataset directory at `path`.
        2. Copy the `fasta_file` into this directory.
        3. Initialize `ExternalData` (which may involve creating a cache).
        4. Build Bowtie2 index for the FASTA file.
        5. Run Jellyfish to count k-mers from the FASTA file.
        6. Create a `dataset.json` file with the dataset definition.

        Args:
            path: The directory where the dataset will be created.
            fasta_file: Path to the input FASTA file.
            species: The species name for this dataset.
            overwrite: If True, allows overwriting existing files (e.g., Jellyfish output).
                Defaults to False.

        Returns:
            An instance of the `Dataset` class.
        """
        path = Path(path)
        fasta_file = Path(fasta_file)

        logger.info(f"Creating dataset at {path}")
        path.mkdir(exist_ok=True, parents=True)
        new_path = path / fasta_file.name
        logger.info(f"Copying {fasta_file} to {new_path}")
        if fasta_file.resolve() != new_path.resolve():
            shutil.copy(fasta_file, new_path)
        del fasta_file

        external_data = ExternalData(
            cache=path / new_path.with_suffix(".parquet").name,
            fasta=new_path,
            regen_cache=overwrite,
        )
        external_data.bowtie_build(overwrite=overwrite)
        external_data.run_jellyfish(overwrite=overwrite)

        Path(path / "dataset.json").write_text(
            DatasetDefinition(
                external_data={
                    "default": ExternalDataDefinition(
                        fasta_name=new_path.name,
                        bowtie2_index_name=external_data.bowtie2_index.name,
                        kmer18_name=external_data.kmer.name,
                    )
                },
                species=species,
            ).model_dump_json()
        )

        return cls(
            path=path,
            external_data=external_data,
            species=species,
            kmer18_path=external_data.kmer,
        )

    @classmethod
    def from_folder(cls, path: Path):
        """
        Loads an existing dataset from a specified folder.

        The folder must contain a `dataset.json` file that defines the dataset structure
        and references the necessary data files (FASTA, k-mer counts, etc.).

        Args:
            path: The path to the dataset folder.

        Returns:
            An instance of the `Dataset` class.

        Raises:
            FileNotFoundError: If `dataset.json` is not found in the specified path.
        """
        path = Path(path)
        if not (path / "dataset.json").exists():
            raise FileNotFoundError(f"Path {path} does not exist. Please create a dataset first.")

        definition = DatasetDefinition.model_validate_json((path / "dataset.json").read_text())
        external_data = ExternalData.from_definition(path, definition.external_data["default"])
        return cls(
            path=path,
            external_data=external_data,
            species=definition.species,
            kmer18_path=path / external_data.kmer,
        )

    def check_kmers(self, seq: str):
        """
        Checks if any 18-mers from the input sequence are present in the `trna_rna_kmers` set.

        This is typically used to filter out sequences that might originate from
        tRNAs or rRNAs, based on a pre-compiled set of common kmers from these RNA types.

        Args:
            seq: The nucleotide sequence to check.

        Returns:
            True if any 18-mer from the sequence is found in `self.trna_rna_kmers`,
            False otherwise. Logs a warning if `self.trna_rna_kmers` is not set.
        """
        if not self.trna_rna_kmers:
            logger.warning("No tRNA-RNA kmers found. Skipping.")
            return False

        return any(x in self.trna_rna_kmers for x in kmers(seq, 18))

    @property
    @cache
    def appris(self):
        raise NotImplementedError("appris not implemented")


class ReferenceDataset(Dataset):
    """
    Represents a pre-configured reference dataset, typically for human or mouse.

    This class inherits from `Dataset` and is specialized for known reference
    genomes/transcriptomes where specific file names and structures are expected
    (e.g., gencode.gtf.gz, ensembl.gtf.gz, appris_data.principal.txt).

    Intended Use:
        - Working with standardized human or mouse reference data provided by
          the fishtools package or a similar pre-packaged dataset.
        - Accessing specific annotation files like APPRIS data.

    Potential Pitfalls:
        - The constructor expects a specific directory structure and file names
          within the provided `path`. If these files are missing or named differently,
          `FileNotFoundError` will be raised.
        - Currently, only "human" and "mouse" species are explicitly supported,
          and using other species names will raise a `ValueError`.
        - If APPRIS data (`appris_data.principal.txt`) is not found, the `appris`
          property will return `None` and log a warning.
    """

    def __init__(self, path: Path | str):
        """
        Initializes a ReferenceDataset object.

        This constructor assumes a specific directory structure and file naming convention
        within the `path` for standard reference data (e.g., GENCODE GTF, Ensembl GTF,
        precomputed k-mer files).

        Args:
            path: The base directory path for the reference dataset.
                The name of this directory is used to infer the species (e.g., "human", "mouse").

        Raises:
            FileNotFoundError: If the `path` does not exist or if essential files
                (e.g., gencode.gtf.gz for human/mouse) are missing.
            ValueError: If the inferred species is not "human" or "mouse".
        """
        self.path = path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        super().__init__(
            path,
            external_data=ExternalData(
                cache=self.path / "gencode.parquet",
                gtf_path=self.path / "gencode.gtf.gz",
                fasta=self.path / "cdna_ncrna_trna.fasta",
                fasta_key_func=lambda x: x.split(" ")[0].split(".")[0],
            ),
            kmer18_path=self.path / "cdna18.jf",
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
        """
        Loads APPRIS principal isoform data from 'appris_data.principal.txt'.

        APPRIS (Annotation of Principal Splice Isoforms) data helps identify the
        main functional transcript(s) for a gene. This method attempts to load
        this data from a standard file name within the dataset path.

        The expected file format is a tab-separated file with columns:
        gene_name, gene_id, transcript_id, ccds, annotation.

        Returns:
            A Polars DataFrame containing the APPRIS data if the file is found
            and successfully parsed. Returns `None` and logs a warning if the
            file is not found.
        """
        try:
            return pl.read_csv(
                self.path / "appris_data.principal.txt",
                separator="\t",
                has_header=False,
                new_columns=[
                    "gene_name",
                    "gene_id",
                    "transcript_id",
                    "ccds",
                    "annotation",
                ],
            )
        except FileNotFoundError:
            logger.warning("No APPRIS data found.")
            return None


@click.command()
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--fasta", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--species", "-s", type=str, help="Species name for metadata")
@click.option("--overwrite", is_flag=True, help="Overwrite existing dataset")
def create_dataset(path: Path, fasta: Path, species: str, overwrite: bool):
    if not species:
        raise ValueError("Species name is required.")
    Dataset.from_components(path, fasta, species=species, overwrite=overwrite)
