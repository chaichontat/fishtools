from .mkprobes.ext.external_data import Dataset
from .mkprobes.utils._alignment import gen_fasta, gen_fastq
from .mkprobes.utils._crawler import crawler
from .mkprobes.utils.samframe import SAMFrame
from .mkprobes.utils.seqcalc import hp, tm
from .mkprobes.utils.sequtils import reverse_complement as rc

IMWRITE_KWARGS = dict(compression=22610, imagej=True, compression_args={"level": 0.75})
