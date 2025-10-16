import sys
import warnings
from typing import TYPE_CHECKING, Dict, Tuple

from .utils.utils import make_lazy_getattr

IMWRITE_KWARGS = dict(compression=22610, bigtiff=True, compressionargs={"level": 0.65})


if sys.platform != "linux":
    warnings.warn(
        "fishtools is only supported on Linux systems. Certain dependencies may not install properly in other platforms.",
        RuntimeWarning,
    )

# Lazy namespace exports (PEP 562)
# name -> (module, attribute)

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # IO / analysis
    "fishread": ("fishtools.analysis.io", "fishread"),
    "metadata": ("fishtools.analysis.io", "metadata"),
    # Core IO
    "Codebook": ("fishtools.io.codebook", "Codebook"),
    "Workspace": ("fishtools.io.workspace", "Workspace"),
    # Probe design utilities
    "Dataset": ("fishtools.mkprobes.ext.dataset", "Dataset"),
    "ReferenceDataset": ("fishtools.mkprobes.ext.dataset", "ReferenceDataset"),
    "gen_fasta": ("fishtools.mkprobes.utils._alignment", "gen_fasta"),
    "gen_fastq": ("fishtools.mkprobes.utils._alignment", "gen_fastq"),
    "crawler": ("fishtools.mkprobes.utils._crawler", "crawler"),
    "SAMFrame": ("fishtools.mkprobes.utils.samframe", "SAMFrame"),
    "hp": ("fishtools.mkprobes.utils.seqcalc", "hp"),
    "tm": ("fishtools.mkprobes.utils.seqcalc", "tm"),
    "gen_idt": ("fishtools.mkprobes.utils.sequtils", "gen_idt"),
    "rc": ("fishtools.mkprobes.utils.sequtils", "reverse_complement"),
    # Pretty printing utilities
    "jprint": ("fishtools.utils.pretty_print", "jprint"),
    "printc": ("fishtools.utils.pretty_print", "printc"),
    "progress_bar": ("fishtools.utils.pretty_print", "progress_bar"),
}

if TYPE_CHECKING:
    # These imports are for static type checkers and IDEs only.
    from fishtools.analysis.io import fishread as fishread
    from fishtools.analysis.io import metadata as metadata
    from fishtools.io.codebook import Codebook as Codebook
    from fishtools.io.workspace import Workspace as Workspace
    from fishtools.mkprobes.ext.dataset import Dataset as Dataset
    from fishtools.mkprobes.ext.dataset import ReferenceDataset as ReferenceDataset
    from fishtools.mkprobes.utils._alignment import gen_fasta as gen_fasta
    from fishtools.mkprobes.utils._alignment import gen_fastq as gen_fastq
    from fishtools.mkprobes.utils._crawler import crawler as crawler
    from fishtools.mkprobes.utils.samframe import SAMFrame as SAMFrame
    from fishtools.mkprobes.utils.seqcalc import hp as hp
    from fishtools.mkprobes.utils.seqcalc import tm as tm
    from fishtools.mkprobes.utils.sequtils import gen_idt as gen_idt
    from fishtools.mkprobes.utils.sequtils import reverse_complement as rc
    from fishtools.utils.pretty_print import jprint as jprint
    from fishtools.utils.pretty_print import printc as printc
    from fishtools.utils.pretty_print import progress_bar as progress_bar


__getattr__, __dir__, __all__ = make_lazy_getattr(globals(), _LAZY_ATTRS, extras=("IMWRITE_KWARGS",))
