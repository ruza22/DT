#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from mrmr import mrmr_classif

#%%
# returns data as an already concatenated numpy matrices
# returns indices for splitting X into individual omics datasets
# X = n x p matrix (n = number of samples, p = number of features)
# Y = n x 3 matrix:
#	1. column corresponds to disease target values
#	2. column corresponds to risk target values
#	3. column corresponds to mutation target values
# norm parameter = True if X is normalized

class Baseline:

	'''
	- loads data into object attributes
	- only complete data are preserved (sample ID appears in all datasets)
	- creates attributes for each dataset, type = pd.DataFrame, rows = features, cols = samples:
		self.allRNA
		self.circRNA
		self.miRNA
		self.piRNA
		self.TE

	- creates attribute self.annot, type = pd.Dataframe, each row is one sample
		- cols "1 disease", "2 risk", "3 mutation" are encoding classification task

	- creates self.X containing features values, type = np.ndarray
	- creates self.Y containing labels, type = np.ndarray

	- creates self.splits -> list of tuples, each tuple contains (start_column_idx, stop_column_idx) of respective omic type in self.X
		- will be used by other object methods to perform baseline tests for specific chosen omic types

	- parameters:
		- norm: True if feature values in self.X are normalized
	'''
	def load_data(self, norm = True):
		# load single type data
		self.allRNA = pd.read_csv("data/200625_allRNA_fromRNAseq_annot_hg38.tsv", sep="\t")
		self.circRNA = pd.read_csv("data/200625_circRNA_fromRNAseq_annot_hg19.tsv", sep="\t")
		self.circRNA.fillna(0, inplace = True)
		self.miRNA = pd.read_excel("data/final_all_samples_miRNA_seq.xlsx")
		self.piRNA = pd.read_excel("data/piRNA_counts.xlsx")
		self.TE = pd.read_csv("data/TE_counts.csv")
		self.annot = pd.read_excel("data/sample sheet for CVUT.xlsx")

		# get sample IDs appearing in each dataset
		id_allRNA = [i.split("_")[0] for i in self.allRNA.columns[6:]]
		id_circRNA = [i.split("_")[0] for i in self.circRNA.columns[9:]]
		id_miRNA = [i.split("_")[0] for i in self.miRNA.columns[1:]]
		id_piRNA = [i.split("_")[0] for i in self.piRNA.columns[1:]]
		id_TE = [i.split("_")[0] for i in self.TE.columns[1:]]
		id_annot = [i.split("_")[0] for i in self.annot.SAMPLE_NAME]

		# clean (parse) ID names
		for i in range(len(id_allRNA)):
			self.allRNA.rename(columns = {self.allRNA.columns[i+6]: id_allRNA[i]}, inplace = True)
		for i in range(len(id_circRNA)):
			self.circRNA.rename(columns = {self.circRNA.columns[i+9]: id_circRNA[i]}, inplace = True)
		for i in range(len(id_miRNA)):
			self.miRNA.rename(columns = {self.miRNA.columns[i+1]: id_miRNA[i]}, inplace = True)
		for i in range(len(id_piRNA)):
			self.piRNA.rename(columns = {self.piRNA.columns[i+1]: id_piRNA[i]}, inplace = True)
		for i in range(len(id_TE)):
			self.TE.rename(columns = {self.TE.columns[i+1]: id_TE[i]}, inplace = True)
		self.annot.SAMPLE_NAME = id_annot


		# create list of all the sample IDs which are present in all datasets
		complete_id = []
		for i in id_allRNA:
			if i in id_miRNA and i in id_piRNA and i in id_circRNA and i in list(self.annot.SAMPLE_NAME):
				complete_id.append(i)

		# modify each individual omic-type dataset to contain only data of complete IDs + annot dataset
		for c in self.allRNA.columns[6:]:
			if c not in complete_id:
				self.allRNA.drop(c, axis = 1, inplace = True)
		for c in self.circRNA.columns[9:]:
			if c not in complete_id:
				self.circRNA.drop(c, axis = 1, inplace = True)
		for c in self.miRNA.columns[1:]:
			if c not in complete_id:
				self.miRNA.drop(c, axis = 1, inplace = True)
		for c in self.piRNA.columns[1:]:
			if c not in complete_id:
				self.piRNA.drop(c, axis = 1, inplace = True)
		for c in self.TE.columns[1:]:
			if c not in complete_id:
				self.TE.drop(c, axis = 1, inplace = True)
		to_delete = []
		for i, s in enumerate(self.annot.SAMPLE_NAME):
			if s not in complete_id:
				to_delete.append(i)
		self.annot.drop(to_delete, axis = 0, inplace = True)

		# concatenating all single-type dataframes into self.X + labels in self.Y
		complete_data = np.zeros((len(complete_id), self.allRNA.shape[0]+self.circRNA.shape[0]+self.miRNA.shape[0]+self.piRNA.shape[0]+self.TE.shape[0] + 3))
		for i, p in enumerate(complete_id):
			allRNA_data = self.allRNA[p].to_numpy()
			circRNA_data = self.circRNA[p].to_numpy()
			miRNA_data = self.miRNA[p].to_numpy()
			piRNA_data = self.piRNA[p].to_numpy()
			TE_data = self.TE[p].to_numpy()
			annot_data = self.annot.loc[self.annot['SAMPLE_NAME'] == complete_id[i]]
			annot_data = annot_data.iloc[:, 2:5].to_numpy()[0]
			row = np.concatenate((allRNA_data, circRNA_data, miRNA_data, piRNA_data, TE_data, annot_data))
			complete_data[i, :] = row

		if norm:
			self.X = normalize(complete_data[:, :-3], axis = 0)
		else:
			self.X = complete_data[:, :-3]
		self.Y = complete_data[:, -3:]

		idx = 0
		self.splits = []
		for dataset in (self.allRNA, self.circRNA, self.miRNA, self.piRNA, self.TE):
			self.splits.append((idx, idx+dataset.shape[0]))
			idx += dataset.shape[0]

	'''
	- performs baseline testing for specified omics types
	- models used for testing:
		1. Logistic regression model with L1 regularization
		2. SVM with linear kernel
		3. XGB classifier

	- parameters:
		- omics: specifies omic types to be included in tested dataset
			- type: string of list of strings
			- if not given, then all are included
			- otherwise valid values are: "allRNA", "circRNA", "miRNA", "piRNA", "TE"
		- C_vals: iverse of regularization strength -> smaller values mean stronger regularization
		- n_select_svm: number of features selected for SVM
		- n_select_xgb: number of features selected for XGB
	'''

	def get_baseline(self, C_vals = [0.42, 0.6, 0.8, 1], n_select_svm = [5, 10, 20, 100],
			     n_select_xgb = [5, 10, 20, 100], omics = "all"):
		split_idx = {"allRNA": 0, "circRNA": 1, "miRNA": 2, "piRNA": 3, "TE": 4}
		if type(omics) == str:
			omics = [omics]
		if omics == ["all"]:
			X = self.X
		else:
			datasets = []
			for o in omics:
				assert o in split_idx.keys(), "Not valid omic name"
				i = split_idx[o]
				datasets.append(self.X[:, self.splits[i][0]:self.splits[i][1]])
			X = datasets[0]
			for i in range(1, len(datasets)):
				X = np.concatenate((X, datasets[i]), axis = 1)
		Y = self.Y
		Y_merge = Y.copy()
		Y_merge[np.where(Y_merge[:, 0] == 1), 0] = 0
		Y_merge[np.where(Y_merge[:, 0] == 2), 0] = 1
		Y_merge[np.where(Y_merge[:, 1] != 0), 1] = 1
		Y_merge[np.where(Y_merge[:, 2] != 0), 2] = 1
				
		CR_d_LR = []
		CR_r_LR = []
		CR_m_LR = []
		fig_LR, axs_LR = plt.subplots(len(C_vals), 3)
		fig_LR.set_figheight(5*len(C_vals))
		fig_LR.set_figwidth(15)
		fig_LR.suptitle("Logistic regression")
		
		for i, c in enumerate(C_vals):
 			LR = LogisticRegression(penalty = "l1", C = c, tol = 0.01, solver = "saga", max_iter = 200)
 			probs_d = cross_val_predict(LR, X, Y[:, 0], cv = 6, method = "predict_proba")
 			probs_r = cross_val_predict(LR, X, Y[:, 1], cv = 6, method = "predict_proba")
 			probs_m = cross_val_predict(LR, X, Y[:, 2], cv = 6, method = "predict_proba")
 			yhat_d = np.argmax(probs_d, axis = 1)
 			yhat_r = np.argmax(probs_r, axis = 1)
 			yhat_m = np.argmax(probs_m, axis = 1)
 			CR_d_LR.append(classification_report(Y[:, 0], yhat_d, zero_division = 0))
 			CR_r_LR.append(classification_report(Y[:, 1], yhat_r, zero_division = 0))
 			CR_m_LR.append(classification_report(Y[:, 2], yhat_m, zero_division = 0))
 			
 			RocCurveDisplay.from_predictions(Y_merge[:, 0], probs_d[:, 1], name = f"C = {c}", ax = axs_LR[i, 0])
 			axs_LR[i, 0].title.set_text("Disease")
 			RocCurveDisplay.from_predictions(Y_merge[:, 1], np.sum(probs_r[:, 1:], axis = 1), name = f"C = {c}", ax = axs_LR[i, 1])
 			axs_LR[i, 1].title.set_text("Risk")
 			RocCurveDisplay.from_predictions(Y_merge[:, 2], np.sum(probs_m[:, 1:], axis = 1), name = f"C = {c}", ax = axs_LR[i, 2])
 			axs_LR[i, 2].title.set_text("Mutation")
		plt.show()
		
		for i, c in enumerate(C_vals):
 			print("Logistic regression:")
 			print(f"---------\nC = {c}\n---------")
 			print("Disease:")
 			print(CR_d_LR[i])
 			print("#########################################")
 			print("Risk:")
 			print(CR_r_LR[i])
 			print("#########################################")
 			print("Mutation:")
 			print(CR_m_LR[i])
 			print("#########################################")
# 			
		CR_d_SVM = []
		CR_r_SVM = []
		CR_m_SVM = []
		fig_SVM, axs_SVM = plt.subplots(len(n_select_svm), 3)
		fig_SVM.set_figheight(5*len(n_select_svm))
		fig_SVM.set_figwidth(15)
		fig_SVM.suptitle("SVM")
		
		SVM = SVC(kernel = "linear", C = 1.0)
		SVM_P = SVC(kernel = "linear", C = 1.0, probability = True)
		
		for i, n in enumerate(n_select_svm):
 			selector = RFE(SVM, n_features_to_select = n, step = 500)
 			
 			selector = selector.fit(X, Y[:, 0])
 			selected = np.where(selector.support_ == True)[0]
 			X_d = X[:, selected]
 			selector = selector.fit(X, Y[:, 1])
 			selected = np.where(selector.support_ == True)[0]
 			X_r = X[:, selected]
 			selector = selector.fit(X, Y[:, 2])
 			selected = np.where(selector.support_ == True)[0]
 			X_m = X[:, selected]
 			
 			yhat_d = cross_val_predict(SVM, X_d, Y[:, 0], cv = 6)
 			yhat_r = cross_val_predict(SVM, X_r, Y[:, 1], cv = 6)
 			yhat_m = cross_val_predict(SVM, X_m, Y[:, 2], cv = 6)
 			probs_d = cross_val_predict(SVM_P, X_d, Y[:, 0], cv = 6, method = "predict_proba")
 			probs_r = cross_val_predict(SVM_P, X_d, Y[:, 1], cv = 6, method = "predict_proba")
 			probs_m = cross_val_predict(SVM_P, X_d, Y[:, 2], cv = 6, method = "predict_proba")
 			CR_d_SVM.append(classification_report(Y[:, 0], np.round(yhat_d), zero_division = 0))
 			CR_r_SVM.append(classification_report(Y[:, 1], np.round(yhat_r), zero_division = 0))
 			CR_m_SVM.append(classification_report(Y[:, 2], np.round(yhat_m), zero_division = 0))
 			
 			RocCurveDisplay.from_predictions(Y_merge[:, 0], probs_d[:, 1], name = f"n = {n}", ax = axs_SVM[i, 0])
 			axs_SVM[i, 0].title.set_text("Disease")
 			RocCurveDisplay.from_predictions(Y_merge[:, 1], np.sum(probs_r[:, 1:], axis = 1), name = f"n = {n}", ax = axs_SVM[i, 1])
 			axs_SVM[i, 1].title.set_text("Risk")
 			RocCurveDisplay.from_predictions(Y_merge[:, 2], np.sum(probs_m[:, 1:], axis = 1), name = f"n = {n}", ax = axs_SVM[i, 2])
 			axs_SVM[i, 2].title.set_text("Mutation")
		plt.show()
			 		
		for i, n in enumerate(n_select_svm):
 			print("SVM:")
 			print(f"---------\nFeatures selected = {n}\n---------")
 			print("Disease:")
 			print(CR_d_SVM[i])
 			print("#########################################")
 			print("Risk:")
 			print(CR_r_SVM[i])
 			print("#########################################")
 			print("Mutation:")
 			print(CR_m_SVM[i])
 			print("#########################################")
			
		CR_d_XGB = []
		CR_r_XGB = []
		CR_m_XGB = []
		fig_XGB, axs_XGB = plt.subplots(len(n_select_xgb), 3)
		fig_XGB.set_figheight(5*len(n_select_xgb))
		fig_XGB.set_figwidth(15)
		fig_XGB.suptitle("GBC")
		
		XGB = GradientBoostingClassifier(max_depth = 8, learning_rate = 1.0)
		
		for i, n in enumerate(n_select_xgb):
 			XGB.fit(X, Y[:, 0])
 			idx = np.argpartition(XGB.feature_importances_, -n)[-n:]
 			X_d = X[:, idx]
 			yhat_d = cross_val_predict(XGB, X_d, Y[:, 0], cv = 6)
 			probs_d = cross_val_predict(XGB, X_d, Y[:, 0], cv = 6, method = "predict_proba")
 			XGB.fit(X, Y[:, 1])
 			idx = np.argpartition(XGB.feature_importances_, -n)[-n:]
 			X_r = X[:, idx]
 			yhat_r = cross_val_predict(XGB, X_r, Y[:, 1], cv = 6)
 			probs_r = cross_val_predict(XGB, X_r, Y[:, 1], cv = 6, method = "predict_proba")
 			XGB.fit(X, Y[:, 2])
 			idx = np.argpartition(XGB.feature_importances_, -n)[-n:]
 			X_m = X[:, idx]
 			yhat_m = cross_val_predict(XGB, X_m, Y[:, 2], cv = 6)
 			probs_m = cross_val_predict(XGB, X, Y[:, 2], cv = 6, method = "predict_proba")
 			CR_d_XGB.append(classification_report(Y[:, 0], yhat_d, zero_division = 0))
 			CR_r_XGB.append(classification_report(Y[:, 1], yhat_r, zero_division = 0))
 			CR_m_XGB.append(classification_report(Y[:, 2], yhat_m, zero_division = 0))
 			
 			RocCurveDisplay.from_predictions(Y_merge[:, 0], probs_d[:, 1], name = f"n = {n}", ax = axs_XGB[i, 0])
 			axs_XGB[i, 0].title.set_text("Disease")
 			RocCurveDisplay.from_predictions(Y_merge[:, 1], np.sum(probs_r[:, 1:], axis = 1), name = f"n = {n}", ax = axs_XGB[i, 1])
 			axs_XGB[i, 1].title.set_text("Risk")
 			RocCurveDisplay.from_predictions(Y_merge[:, 2], np.sum(probs_m[:, 1:], axis = 1), name = f"n = {n}", ax = axs_XGB[i, 2])
 			axs_XGB[i, 2].title.set_text("Mutation")
		plt.show()
			 
			 
 			
		for i, n in enumerate(n_select_xgb):
 			print("XGB:")
 			print(f"---------\nFeatures selected = {n}\n---------")
 			print("Disease:")
 			print(CR_d_XGB[i])
 			print("#########################################")
 			print("Risk:")
 			print(CR_r_XGB[i])
 			print("#########################################")
 			print("Mutation:")
 			print(CR_m_XGB[i])
 			print("#########################################")