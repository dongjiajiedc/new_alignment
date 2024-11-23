library(tidyverse)
library(dyngen)
library(zellkonverter)
library(Matrix)
library(MASS)


set.seed(1)
realcount <-readH5AD('./n.h5ad')
realcount <- realcount@assays@data$X
realcount <- Matrix(realcount, sparse = TRUE)

backbone <- backbone_linear()
num_cells <- 500
num_feats <- ncol(realcount)
num_tfs <- nrow(backbone$module_info)
num_tar <- round((num_feats - num_tfs) / 2)
num_hks <- num_feats - num_tfs - num_tar

config <-
  initialise_model(
    backbone = backbone,
    num_cells = num_cells,
    num_tfs = num_tfs,
    num_targets = num_tar,
    num_hks = num_hks,
    verbose = interactive(),
    download_cache_dir = tools::R_user_dir("dyngen", "data"),
    simulation_params = simulation_default(
      total_time = 1000,
      census_interval = 2, 
      ssa_algorithm = ssa_etl(tau = 300/3600),
      experiment_params = simulation_type_wild_type(num_simulations = 10)
    ),
    experiment_params = experiment_snapshot(
      realcount = realcount
    )
  )
out <- generate_dataset(config, make_plots = TRUE)

datasets <- list(
  real = t(as.matrix(realcount)),
  dyngen = t(as.matrix(out$dataset$counts))
)

ddsList <- lapply(datasets, function(ds) {
  DESeq2::DESeqDataSetFromMatrix(
    countData = round(as.matrix(ds)), 
    colData = data.frame(sample = seq_len(ncol(ds))), 
    design = ~1
  )
})
write.table(ddsList$dyngen@assays@data$counts,file="500n.csv")
ddsList$dyngen@assays@data$counts