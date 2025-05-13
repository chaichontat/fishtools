import sys
import warnings

from .analysis.io import fishread, metadata
from .mkprobes.ext.dataset import Dataset, ReferenceDataset
from .mkprobes.utils._alignment import gen_fasta, gen_fastq
from .mkprobes.utils._crawler import crawler
from .mkprobes.utils.samframe import SAMFrame
from .mkprobes.utils.seqcalc import hp, tm
from .mkprobes.utils.sequtils import gen_idt
from .mkprobes.utils.sequtils import reverse_complement as rc
from .utils.pretty_print import jprint, printc, progress_bar

IMWRITE_KWARGS = dict(compression=22610, imagej=True, compression_args={"level": 0.65})


if sys.platform != "linux":
    warnings.warn(
        "fishtools is only supported on Linux systems. Certain dependencies may not install properly in other platforms.",
        RuntimeWarning,
    )
