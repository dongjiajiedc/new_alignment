setwd('/Users/dongjiajie/Desktop/alignment/fyp/hyperalignment/eval/scMerge')
suppressPackageStartupMessages({
library(SingleCellExperiment)
library(scMerge)
library(scater)
library(zellkonverter)

})



adata <- readH5AD('../datas/32/adata12.h5ad')

adata@assays@data@listData[["counts"]] <- array(as.vector(adata@assays@data@listData[["X"]]), dim = dim(adata@assays@data@listData[["X"]]))

adata@assays@data@listData[["logcounts"]] <- array(as.vector(adata@assays@data@listData[["X"]]), dim = dim(adata@assays@data@listData[["X"]]))

scMerge_unsupervised <- scMerge(
  sce_combine = adata, 
  ctl = rownames(adata@assays@data@listData[["X"]]),
  kmeansK =  c(5,5),
  assay_name = "scMerge_unsupervised")

writeH5AD(scMerge_unsupervised,'after.h5ad',X='scMerge_unsupervised')


