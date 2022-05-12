library(Matrix)
library(reticulate)
anndata <- import('anndata')

## Read RDS objects
data_dir <- 'GASPACHO/Data/'
data_file <- paste0(data_dir, 'log_cpm_4999_22188.RDS')
data <- readRDS(data_file)
metadata <- readRDS(list.files(data_dir, pattern = 'metadata', full.names = TRUE))
init_params <- readRDS(list.files(data_dir, pattern = 'init', full.names = TRUE))

## Store in AnnData objects
adata <- anndata$AnnData(X = t(data), obs=metadata)
init_x = cbind(init_params$Xi[[1]], init_params$Xi[[2]])
adata$obsm$update(X_init = init_x)

## Save as .h5ad
adata$write_h5ad(paste0(data_dir, 'ipsc_scRNA.h5ad'))