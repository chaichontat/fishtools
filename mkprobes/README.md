# mkprobes

`mkprobes` contains the combinatorial FISH probe-design code currently shipped
inside `fishtools`. This nested project is the extraction target for a future
monorepo; the legacy `fishtools.mkprobes` package remains unchanged during the
initial migration.

The Python package requires Python 3.12 or newer. Full reference preparation
and screening workflows also require Bowtie 2, Jellyfish, MAFFT, and
RepeatMasker on `PATH`.

The first extraction excludes the experimental `picker` and `readouts`
notebooks because they execute project-specific work at import time and depend
on stale modules.

Project-specific panel generation and assembly drivers are retained under
`scripts/`. They are distributed in the source archive, not installed as
console commands.
