#%%
from mogonet_data_load import *
from baseline import Baseline
#%%
bl = Baseline()
bl.load_data()
#%%
# get_train_test_idx()
#%%
# get_mogonet_labels(bl.Y)
#%%
# get_mogonet_datasets_allRNA(bl.X, bl.Y, bl.splits, "MUTATION")
# get_mogonet_datasets_circRNA(bl.X, bl.Y, bl.splits, "MUTATION")
# get_mogonet_datasets_miRNA(bl.X, bl.Y, bl.splits, "MUTATION")
# get_mogonet_datasets_piRNA(bl.X, bl.Y, bl.splits, "MUTATION")
# get_mogonet_datasets_TE(bl.X, bl.Y, bl.splits, "MUTATION")