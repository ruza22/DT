#%%
import pandas as pd
import numpy as np
from mrmr import mrmr_classif
from os import path

#%%
def get_train_test_idx():
	idx = np.random.choice(66, 66, replace = False)
	idx_test = idx[:16]
	idx_train = idx[16:]
	with open("MOGONET/index/tr.txt", "w") as f:
		for i in idx_train:
			f.write(str(i))
			f.write("\n")
	with open("MOGONET/index/te.txt", "w") as f:
		for i in idx_test:
			f.write(str(i))
			f.write("\n")
#%%
def get_mogonet_datasets_allRNA(X, Y, s, dataset):
	assert dataset in ["DISEASE", "RISK", "MUTATION"], "Wrong dataset name"
	if dataset == "DISEASE":
		Y = Y[:, 0]
		data_path = "MOGONET/DISEASE"
	elif dataset == "RISK":
		Y = Y[:, 1]
		data_path = "MOGONET/RISK"
	elif dataset == "MUTATION":
		Y = Y[:, 2]
		data_path = "MOGONET/MUTATION"
	
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
	
	# allRNA dataset is too large to be allocated in memory
	# allRNA dataset didived into halves -> from each 500 features are selected
	# then from those 1000 features final 200 is selected again using MRMR
	X_allRNA = X[:, s[0][0]:s[0][1]]
	half_idx = int(X_allRNA.shape[1]/2)
	X_allRNA_1 = X_allRNA[:, 0:half_idx]
	X_allRNA_2 = X_allRNA[:, half_idx:]
	
	allRNA_preselect = []
 	
	i = 0
	for d in (X_allRNA_1, X_allRNA_2):
		d = pd.DataFrame(d)
		f_select = mrmr_classif(X=d, y=Y, K=500)
		for f in f_select:
			if i == 1:
				allRNA_preselect.append(f+half_idx)
			else:
				allRNA_preselect.append(f)
		i+=1
	X_allRNA = X_allRNA[:, allRNA_preselect]
	allRNA_200 = mrmr_classif(X=pd.DataFrame(X_allRNA), y=Y, K=200)
	X_allRNA = X_allRNA[:, allRNA_200]
	
	pd.DataFrame(X_allRNA[rand_tr, :]).to_csv(path.join(data_path, "1_tr.csv"), index = False, header = False)
	pd.DataFrame(X_allRNA[rand_te, :]).to_csv(path.join(data_path, "1_te.csv"), index = False, header = False)
	
	allRNA_raw = pd.read_csv("data/200625_allRNA_fromRNAseq_annot_hg38.tsv", sep = "\t")
	allRNA_preselect_fnames = allRNA_raw.GENE_NAME[allRNA_preselect]
	allRNA_fnames = allRNA_preselect_fnames.iloc[allRNA_200]
	allRNA_fnames.to_csv(path.join(data_path, "1_featname.csv"), index = False, header = False)
	
def get_mogonet_datasets_circRNA(X, Y, s, dataset):
	assert dataset in ["DISEASE", "RISK", "MUTATION"], "Wrong dataset name"
	if dataset == "DISEASE":
		Y = Y[:, 0]
		data_path = "MOGONET/DISEASE"
	elif dataset == "RISK":
		Y = Y[:, 1]
		data_path = "MOGONET/RISK"
	elif dataset == "MUTATION":
		Y = Y[:, 2]
		data_path = "MOGONET/MUTATION"
		
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
		
	X_circRNA = X[:, s[1][0]:s[1][1]]
	circRNA_200 = mrmr_classif(X=pd.DataFrame(X_circRNA), y=Y, K=200)
	X_circRNA = X_circRNA[:, circRNA_200]
	pd.DataFrame(X_circRNA[rand_tr, :]).to_csv(path.join(data_path, "2_tr.csv"), index = False, header = False)
	pd.DataFrame(X_circRNA[rand_te, :]).to_csv(path.join(data_path, "2_te.csv"), index = False, header = False)
	
	circRNA_raw = pd.read_csv("data/200625_circRNA_fromRNAseq_annot_hg19.tsv", sep="\t")
	circRNA_fnames = circRNA_raw.GENE_NAME[circRNA_200]
	circRNA_fnames.to_csv(path.join(data_path, "2_featname.csv"), index = False, header = False)
	
def get_mogonet_datasets_miRNA(X, Y, s, dataset):
	assert dataset in ["DISEASE", "RISK", "MUTATION"], "Wrong dataset name"
	if dataset == "DISEASE":
		Y = Y[:, 0]
		data_path = "MOGONET/DISEASE"
	elif dataset == "RISK":
		Y = Y[:, 1]
		data_path = "MOGONET/RISK"
	elif dataset == "MUTATION":
		Y = Y[:, 2]
		data_path = "MOGONET/MUTATION"
		
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
	
	X_miRNA = X[:, s[2][0]:s[2][1]]
	miRNA_200 = mrmr_classif(X=pd.DataFrame(X_miRNA), y=Y, K=200)
	X_miRNA = X_miRNA[:, miRNA_200]
	pd.DataFrame(X_miRNA[rand_tr, :]).to_csv(path.join(data_path, "3_tr.csv"), index = False, header = False)
	pd.DataFrame(X_miRNA[rand_te, :]).to_csv(path.join(data_path, "3_te.csv"), index = False, header = False)
	
	miRNA_raw = pd.read_excel("data/final_all_samples_miRNA_seq.xlsx")
	miRNA_fnames = miRNA_raw.miRNA[miRNA_200]
	miRNA_fnames.to_csv(path.join(data_path, "3_featname.csv"), index = False, header = False)
	
def get_mogonet_datasets_piRNA(X, Y, s, dataset):
	assert dataset in ["DISEASE", "RISK", "MUTATION"], "Wrong dataset name"
	if dataset == "DISEASE":
		Y = Y[:, 0]
		data_path = "MOGONET/DISEASE"
	elif dataset == "RISK":
		Y = Y[:, 1]
		data_path = "MOGONET/RISK"
	elif dataset == "MUTATION":
		Y = Y[:, 2]
		data_path = "MOGONET/MUTATION"
		
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
	
	X_piRNA = X[:, s[3][0]:s[3][1]]
	piRNA_200 = mrmr_classif(X=pd.DataFrame(X_piRNA), y=Y, K=200)
	X_piRNA = X_piRNA[:, piRNA_200]
	pd.DataFrame(X_piRNA[rand_tr, :]).to_csv(path.join(data_path, "4_tr.csv"), index = False, header = False)
	pd.DataFrame(X_piRNA[rand_te, :]).to_csv(path.join(data_path, "4_te.csv"), index = False, header = False)
	
	piRNA_raw = pd.read_excel("data/piRNA_counts.xlsx")
	piRNA_fnames = piRNA_raw.piRNA[piRNA_200]
	piRNA_fnames.to_csv(path.join(data_path, "4_featname.csv"), index = False, header = False)
	
def get_mogonet_datasets_TE(X, Y, s, dataset):
	assert dataset in ["DISEASE", "RISK", "MUTATION"], "Wrong dataset name"
	if dataset == "DISEASE":
		Y = Y[:, 0]
		data_path = "MOGONET/DISEASE"
	elif dataset == "RISK":
		Y = Y[:, 1]
		data_path = "MOGONET/RISK"
	elif dataset == "MUTATION":
		Y = Y[:, 2]
		data_path = "MOGONET/MUTATION"
		
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
	
	X_TE = X[:, s[4][0]:s[4][1]]
	TE_200 = mrmr_classif(X=pd.DataFrame(X_TE), y=Y, K=200)
	X_TE = X_TE[:, TE_200]
	pd.DataFrame(X_TE[rand_tr, :]).to_csv(path.join(data_path, "5_tr.csv"), index = False, header = False)
	pd.DataFrame(X_TE[rand_te, :]).to_csv(path.join(data_path, "5_te.csv"), index = False, header = False)
	
	TE_raw = pd.read_csv("data/TE_counts.csv")
	TE_fnames = TE_raw.TE[TE_200]
	TE_fnames.to_csv(path.join(data_path, "5_featname.csv"), index = False, header = False)
	
def get_mogonet_labels(Y):
	rand_tr = np.zeros(50, dtype = int)
	rand_te = np.zeros(16, dtype = int)
	
	with open("MOGONET/index/tr.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_tr[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	with open("MOGONET/index/te.txt", "r") as f:
		i = f.readline().strip()
		j = 0
		while i:
			rand_te[j] = int(i)
			j += 1
			i = f.readline().strip()
			
	labels_d = pd.DataFrame(Y[:, 0])
	labels_d = labels_d.subtract(1)
	labels_d.iloc[rand_te].to_csv("MOGONET/DISEASE/labels_te.csv", index = False, header = False)
	labels_d.iloc[rand_tr].to_csv("MOGONET/DISEASE/labels_tr.csv", index = False, header = False)
	
	labels_r = pd.DataFrame(Y[:, 1])
	labels_r.iloc[rand_te].to_csv("MOGONET/RISK/labels_te.csv", index = False, header = False)
	labels_r.iloc[rand_tr].to_csv("MOGONET/RISK/labels_tr.csv", index = False, header = False)
	
	labels_m = pd.DataFrame(Y[:, 2])
	labels_m.iloc[rand_te].to_csv("MOGONET/MUTATION/labels_te.csv", index = False, header = False)
	labels_m.iloc[rand_tr].to_csv("MOGONET/MUTATION/labels_tr.csv", index = False, header = False)
	
#%%


#%%
# bl.get_mogonet_datasets("MOGONET/MYDATA")
#%%
# bl.get_baseline(omics = ["TE"])

#%%
# fig, ax = plt.subplots(2, 3)
# bl.disp_d.plot(ax[0, 0])
# ax[0, 0].title.set_text("Disease, C = 0.42")
# bl.disp_r.plot(ax[0, 1])
# ax[0, 1].title.set_text("Risk, C = 0.6")
# bl.disp_m.plot(ax[0, 2])
# ax[0, 2].title.set_text("Mutation, C = 0.8")
# bl.disp_d.plot(ax[1, 0])
# ax[1, 0].title.set_text("Disease, C = 0.42")
# bl.disp_r.plot(ax[1, 1])
# ax[1, 1].title.set_text("Risk, C = 0.6")
# bl.disp_m.plot(ax[1, 2])
# ax[1, 2].title.set_text("Mutation, C = 0.8")
# plt.show()

#%%

# due to class imbalance: 7(0), 59(1)
# 4 random (0) and  33 random (1) are selected for train dataset, rest test
# rand0 = np.random.choice(7, 7, replace = False)
# rand0_tr = rand0[:4]
# rand0_te = rand0[4:]
# rand1 = np.random.choice(np.arange(7, 66), 59, replace = False)
# rand1_tr = rand1[:33]
# rand1_te = rand1[33:]

# rand_tr = np.concatenate((rand0_tr, rand1_tr))
# rand_te = np.concatenate((rand0_te, rand1_te))

# s = bl.splits

#%%
# # allRNA dataset is too large to be allocated in memory
# # allRNA dataset didived into halves -> from each 500 features are selected
# # then from those 1000 features final 200 is selected again using MRMR
# X_allRNA = bl.X[:, s[0][0]:s[0][1]]
# half_idx = int(X_allRNA.shape[1]/2)
# X_allRNA_1 = X_allRNA[:, 0:half_idx]
# X_allRNA_2 = X_allRNA[:, half_idx:]

# allRNA_preselect = []

# i = 0
# for d in (X_allRNA_1, X_allRNA_2):
# 	d = pd.DataFrame(d)
# 	f_select = mrmr_classif(X=d, y=bl.Y[:, 0], K=500)
# 	for f in f_select:
# 		if i == 1:
# 			allRNA_preselect.append(f+half_idx)
# 		else:
# 			allRNA_preselect.append(f)
# 	i+=1
# X_allRNA = X_allRNA[:, allRNA_preselect]
# allRNA_200 = mrmr_classif(X=pd.DataFrame(X_allRNA), y=bl.Y[:, 0], K=200)
# X_allRNA = X_allRNA[:, allRNA_200]

# pd.DataFrame(X_allRNA[rand_tr, :]).to_csv("MOGONET/MYDATA/1_tr.csv", index = False, header = False)
# pd.DataFrame(X_allRNA[rand_te, :]).to_csv("MOGONET/MYDATA/1_te.csv", index = False, header = False)

# allRNA_raw = pd.read_csv("data/200625_allRNA_fromRNAseq_annot_hg38.tsv", sep = "\t")
# allRNA_preselect_fnames = allRNA_raw.GENE_NAME[allRNA_preselect]
# allRNA_fnames = allRNA_preselect_fnames.iloc[allRNA_200]
# allRNA_fnames.to_csv("MOGONET/MYDATA/1_featname.csv", index = False, header = False)
	
	
#%%
# X_circRNA = bl.X[:, s[1][0]:s[1][1]]
# circRNA_200 = mrmr_classif(X=pd.DataFrame(X_circRNA), y=bl.Y[:, 0], K=200)
# X_circRNA = X_circRNA[:, circRNA_200]
# pd.DataFrame(X_circRNA[rand_tr, :]).to_csv("MOGONET/MYDATA/2_tr.csv", index = False, header = False)
# pd.DataFrame(X_circRNA[rand_te, :]).to_csv("MOGONET/MYDATA/2_te.csv", index = False, header = False)

# circRNA_raw = pd.read_csv("data/200625_circRNA_fromRNAseq_annot_hg19.tsv", sep="\t")
# circRNA_fnames = circRNA_raw.GENE_NAME[circRNA_200]
# circRNA_fnames.to_csv("MOGONET/MYDATA/2_featname.csv", index = False, header = False)
#%%
# X_miRNA = bl.X[:, s[2][0]:s[2][1]]
# miRNA_200 = mrmr_classif(X=pd.DataFrame(X_miRNA), y=bl.Y[:, 0], K=200)
# X_miRNA = X_miRNA[:, miRNA_200]
# pd.DataFrame(X_miRNA[rand_tr, :]).to_csv("MOGONET/MYDATA/3_tr.csv", index = False, header = False)
# pd.DataFrame(X_miRNA[rand_te, :]).to_csv("MOGONET/MYDATA/3_te.csv", index = False, header = False)

# miRNA_raw = pd.read_excel("data/final_all_samples_miRNA_seq.xlsx")
# miRNA_fnames = miRNA_raw.miRNA[miRNA_200]
# miRNA_fnames.to_csv("MOGONET/MYDATA/3_featname.csv", index = False, header = False)

#%%
# X_piRNA = bl.X[:, s[3][0]:s[3][1]]
# piRNA_200 = mrmr_classif(X=pd.DataFrame(X_piRNA), y=bl.Y[:, 0], K=200)
# X_piRNA = X_piRNA[:, piRNA_200]
# pd.DataFrame(X_piRNA[rand_tr, :]).to_csv("MOGONET/MYDATA/4_tr.csv", index = False, header = False)
# pd.DataFrame(X_piRNA[rand_te, :]).to_csv("MOGONET/MYDATA/4_te.csv", index = False, header = False)

# piRNA_raw = pd.read_excel("data/piRNA_counts.xlsx")
# piRNA_fnames = piRNA_raw.piRNA[piRNA_200]
# piRNA_fnames.to_csv("MOGONET/MYDATA/4_featname.csv", index = False, header = False)
#%%
# X_TE = bl.X[:, s[4][0]:s[4][1]]
# TE_200 = mrmr_classif(X=pd.DataFrame(X_TE), y=bl.Y[:, 0], K=200)
# X_TE = X_TE[:, TE_200]
# pd.DataFrame(X_TE[rand_tr, :]).to_csv("MOGONET/MYDATA/5_tr.csv", index = False, header = False)
# pd.DataFrame(X_TE[rand_te, :]).to_csv("MOGONET/MYDATA/5_te.csv", index = False, header = False)

# TE_raw = pd.read_csv("data/TE_counts.csv")
# TE_fnames = TE_raw.TE[TE_200]
# TE_fnames.to_csv("MOGONET/MYDATA/5_featname.csv", index = False, header = False)

#%%
# labels = pd.DataFrame(bl.Y[:, 0])
# labels = labels.subtract(1)
# labels.iloc[rand_te].to_csv("MOGONET/MYDATA/labels_te.csv", index = False, header = False)
# labels.iloc[rand_tr].to_csv("MOGONET/MYDATA/labels_tr.csv", index = False, header = False)






















