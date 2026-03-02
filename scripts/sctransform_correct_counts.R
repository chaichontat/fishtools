args <- commandArgs(trailingOnly = TRUE)

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript sctransform_correct_counts.R",
    " --counts <counts_gxc.mtx>",
    " --genes <genes.tsv>",
    " --cells <cells.tsv>",
    " --cell-attr <cell_attr.tsv>",
    " --batch-key <obs_column>",
    " --out <corrected_gxc.mtx>",
    " [--verbosity <int>]\n",
    sep = ""
  )
}

parse_args <- function(argv) {
  parsed <- list(
    counts = NULL,
    genes = NULL,
    cells = NULL,
    cell_attr = NULL,
    batch_key = NULL,
    out = NULL,
    verbosity = 2L
  )

  i <- 1L
  n <- length(argv)
  while (i <= n) {
    key <- argv[[i]]
    if (key %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    }
    if (substr(key, 1L, 2L) != "--") {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    if (i == n) {
      stop(sprintf("Missing value for %s", key), call. = FALSE)
    }
    value <- argv[[i + 1L]]
    if (key == "--counts") parsed$counts <- value
    else if (key == "--genes") parsed$genes <- value
    else if (key == "--cells") parsed$cells <- value
    else if (key == "--cell-attr") parsed$cell_attr <- value
    else if (key == "--batch-key") parsed$batch_key <- value
    else if (key == "--out") parsed$out <- value
    else if (key == "--verbosity") parsed$verbosity <- as.integer(value)
    else stop(sprintf("Unknown option: %s", key), call. = FALSE)
    i <- i + 2L
  }

  required <- c("counts", "genes", "cells", "cell_attr", "batch_key", "out")
  for (name in required) {
    if (is.null(parsed[[name]]) || identical(parsed[[name]], "")) {
      stop(sprintf("Missing required option --%s", gsub("_", "-", name)), call. = FALSE)
    }
  }
  parsed
}

opt <- parse_args(args)

suppressPackageStartupMessages({
  library(Matrix)
  library(future)
  library(sctransform)
})

options(future.globals.maxSize = Inf)
future::plan("multisession", workers = 16)
print(future::plan())

counts <- Matrix::readMM(opt$counts)
counts <- as(counts, "CsparseMatrix")
if (!inherits(counts, "dgCMatrix")) {
  counts <- as(counts, "dgCMatrix")
}

genes <- readLines(opt$genes, warn = FALSE)
cells <- readLines(opt$cells, warn = FALSE)

if (nrow(counts) != length(genes)) {
  stop(
    sprintf("nrow(counts)=%d does not match number of genes=%d", nrow(counts), length(genes)),
    call. = FALSE
  )
}
if (ncol(counts) != length(cells)) {
  stop(
    sprintf("ncol(counts)=%d does not match number of cells=%d", ncol(counts), length(cells)),
    call. = FALSE
  )
}

rownames(counts) <- genes
colnames(counts) <- cells

cell_attr <- utils::read.table(
  opt$cell_attr,
  sep = "\t",
  header = TRUE,
  stringsAsFactors = FALSE,
  quote = "",
  comment.char = ""
)

if (!("cell" %in% colnames(cell_attr))) {
  stop("cell_attr.tsv must contain a 'cell' column.", call. = FALSE)
}
if (!(opt$batch_key %in% colnames(cell_attr))) {
  stop(sprintf("cell_attr.tsv is missing batch key column '%s'.", opt$batch_key), call. = FALSE)
}

cell_index <- match(cells, cell_attr$cell)
if (anyNA(cell_index)) {
  stop("cell_attr.tsv does not contain all cells listed in cells.tsv.", call. = FALSE)
}
cell_attr <- cell_attr[cell_index, , drop = FALSE]
rownames(cell_attr) <- cell_attr$cell

if (any(is.na(cell_attr[[opt$batch_key]]))) {
  stop(sprintf("Batch key column '%s' contains missing values.", opt$batch_key), call. = FALSE)
}
if (!("log_umi" %in% colnames(cell_attr))) {
  umi_totals <- Matrix::colSums(counts)
  cell_attr$log_umi <- log10(pmax(umi_totals, 1))
}

fit <- sctransform::vst(
  umi = counts,
  cell_attr = cell_attr,
  latent_var = "log_umi",
  batch_var = opt$batch_key,
  vst.flavor = "v2",
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = opt$verbosity
)

corrected <- sctransform::correct_counts(
  x = fit,
  umi = counts,
  cell_attr = cell_attr,
  verbosity = opt$verbosity
)
corrected <- as(corrected, "dgCMatrix")

Matrix::writeMM(corrected, file = opt$out)
