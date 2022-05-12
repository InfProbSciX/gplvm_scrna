# use reticulate to save the data as scipy sparse matrices

library(reticulate)

data_dir = 'GASPACHO/Data/'

inits = readRDS(paste0(data_dir, 'init_param.RDS'))
init_x = cbind(inits$Xi[[1]], inits$Xi[[2]])
import('numpy')$save('data/init_x.npy', init_x)
import('numpy')$save('data/init_re_mu.npy', inits$zeta[-1])
import('numpy')$save('data/init_re_sg.npy', inits$delta[-1])

metadata = readRDS(paste0(data_dir, 'metadata.RDS'))
data = readRDS(paste0(data_dir, 'log_cpm_4999_22188.RDS'))

write.csv(metadata, 'data/metadata.csv', row.names=F)
model_mat = cbind(
	model.matrix(~ 0 + nfrag + ngene, data=metadata),
	model.matrix(~ 0 + donor, data=metadata),
	model.matrix(~ 0 + mt + ercc, data=metadata),
	model.matrix(~ 0 + plate, data=metadata)
) # cbind here because model.matrix removes a factor from donors/plates
# due to unidentifiability but that is ignored here, and 200 dims are returned

write.csv(model_mat, 'data/model_mat.csv', row.names=F)

# repl_python()
py_run_string(code = '

from scipy.sparse import save_npz
save_npz("data/data.npz", r.data)

')
