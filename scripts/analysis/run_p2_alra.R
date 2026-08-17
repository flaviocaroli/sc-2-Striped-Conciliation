#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)

if (length(args) != 7) {
    stop(
        paste(
            "Usage: run_p2_alra.R",
            "<input_bin> <n_rows> <n_cols>",
            "<output_bin> <metadata_tsv>",
            "<singular_values_csv>",
            "<num_of_sds_csv>"
        )
    )
}

input_bin <- args[[1]]
n_rows <- as.integer(args[[2]])
n_cols <- as.integer(args[[3]])
output_bin <- args[[4]]
metadata_tsv <- args[[5]]
singular_values_csv <- args[[6]]
num_of_sds_csv <- args[[7]]

if (
    is.na(n_rows)
    || is.na(n_cols)
    || n_rows <= 100
    || n_cols <= 100
) {
    stop(
        "ALRA automatic choose_k requires "
        "both dimensions > 100 for K=100."
    )
}

suppressPackageStartupMessages(
    library(ALRA)
)

expected_n <- n_rows * n_cols

values <- readBin(
    input_bin,
    what="numeric",
    n=expected_n,
    size=4,
    endian="little"
)

if (length(values) != expected_n) {
    stop(
        sprintf(
            "Expected %d float32 values, read %d.",
            expected_n,
            length(values)
        )
    )
}

A_norm <- matrix(
    values,
    nrow=n_rows,
    ncol=n_cols,
    byrow=TRUE
)

if (!all(is.finite(A_norm))) {
    stop("Input contains non-finite values.")
}

if (any(A_norm < 0)) {
    stop("Input contains negative values.")
}

#
# Frozen protocol:
# one seed, implementation automatic choose_k.
#
# This explicit call exposes chosen k while preserving
# the exact RNG stream of alra(A_norm, k=0).
#
set.seed(20260728)

choice <- ALRA::choose_k(
    A_norm,
    K=100,
    thresh=6,
    noise_start=80,
    q=2,
    use.mkl=FALSE
)

chosen_k <- as.integer(choice$k)

if (
    length(chosen_k) != 1
    || is.na(chosen_k)
    || chosen_k <= 0
) {
    stop("ALRA choose_k returned invalid k.")
}

#
# IMPORTANT:
# do not reset RNG here.
#
result <- ALRA::alra(
    A_norm,
    k=chosen_k,
    q=10,
    quantile.prob=0.001,
    use.mkl=FALSE
)

completed <- result$A_norm_rank_k_cor_sc

if (!identical(dim(completed), dim(A_norm))) {
    stop(
        sprintf(
            "Completed matrix dimension mismatch: %s vs %s",
            paste(dim(completed), collapse="x"),
            paste(dim(A_norm), collapse="x")
        )
    )
}

if (!all(is.finite(completed))) {
    stop("Completed matrix contains non-finite values.")
}

if (min(completed) < 0) {
    stop("Completed matrix contains negative values.")
}

#
# Python expects C-order / row-major float32.
# as.numeric(t(M)) gives row-major traversal of M.
#
con <- file(
    output_bin,
    open="wb"
)

writeBin(
    as.numeric(t(completed)),
    con,
    size=4,
    endian="little"
)

close(con)

metadata <- c(
    paste0("chosen_k\t", chosen_k),
    "random_seed\t20260728",
    "choose_k_K\t100",
    "choose_k_thresh\t6",
    "choose_k_noise_start\t80",
    "choose_k_q\t2",
    "alra_q\t10",
    "quantile_prob\t0.001",
    "use_mkl\tFALSE",
    paste0("n_rows\t", n_rows),
    paste0("n_cols\t", n_cols),
    paste0(
        "input_nonzero_fraction\t",
        mean(A_norm > 0)
    ),
    paste0(
        "completed_nonzero_fraction\t",
        mean(completed > 0)
    ),
    paste0(
        "completed_min\t",
        min(completed)
    ),
    paste0(
        "completed_max\t",
        max(completed)
    )
)

writeLines(
    metadata,
    metadata_tsv
)

write.csv(
    data.frame(
        index=seq_along(choice$d),
        singular_value=choice$d
    ),
    singular_values_csv,
    row.names=FALSE
)

write.csv(
    data.frame(
        index=seq_along(choice$num_of_sds),
        num_of_sds=choice$num_of_sds
    ),
    num_of_sds_csv,
    row.names=FALSE
)

cat(
    sprintf(
        "P2_ALRA_CHOSEN_K=%d\n",
        chosen_k
    )
)

cat(
    "P2_ALRA_R_WRAPPER=PASS\n"
)
