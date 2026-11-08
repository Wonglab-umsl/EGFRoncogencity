#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import mdtraj as md
import pickle
from sklearn import tree
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, accuracy_score

#try setting the seed randomly
seed=None
#number of replicate simulations for each mutant
nreplica=int(sys.argv[1])
#this is the maximum depth of the trees
maxdepth=int(sys.argv[2])
nrepeat=int(sys.argv[3])
input=open('list1','r')
#names=[]
labels={}
for line in input:
	#number, simulation name, label
	if (line[0]!='#'):
		words=line.split()
		if (len(words)>0):
			name=words[1]
			lbl=words[2]
			if (len(words)>3):
				lbl+=' '+words[3]
			labels[name]=lbl

names=labels.keys()

benign_mutants=['egfr-wt-active-charmm','egfr-v904i-active-charmm','egfr-i926n-active-charmm','egfr-i966v-active-charmm','egfr-f712l-active-charmm','egfr-s720f-active-charmm','egfr-p741s-active-charmm','egfr-i744m-active-charmm','egfr-v774l-active-charmm','egfr-r831h-active-charmm']


print('reading secondary structure info',flush=True)
with open('egfr-sec-struc-info.p','rb') as inputf:
	ss=pickle.load(inputf)


#"simplify" the secondary structure DSSP 
#simp_table={'H':'H','G':'H','I':'H','E':'E','B':'E','T':'C','S':'C',' ':'C'}
#ss_simplified={}
#for name in names:
#	ss_simplified[name]=np.vectorize(simp_table.get)(ss[name])

ss_names={'H':'helix','E':'sheet'}
index_labels=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]


psffname='../anal/egfr-wt-active-charmm.psf'.format(name)
top=md.load_psf(psffname)
sel='name CA and residue 705 to 974 and not residue 746 to 759 and not residue 769 to 771 and not residue 773 to 774'
ca_atoms=top.select(sel)
resnums=[top.atom(idx).residue.resSeq for idx in ca_atoms]
ss_col_labels=[f'{resnum:d}-{ss_name}' for resnum in resnums for ss_name in ss_names.values()]
ss_frac_df=pd.DataFrame(columns=ss_col_labels,index=index_labels)



for name in names:
	psffname=f'../anal/{name}.psf'
	top=md.load_psf(psffname)
	ca_atoms=top.select(sel)
	ires_list=[top.atom(idx).residue.index for idx in ca_atoms]
	for id in range(1,nreplica+1):
		for ss_type in ss_names.keys():
			this_ss=ss[(name,id)][:,:]
			frac=np.count_nonzero(this_ss==ss_type,axis=0)/this_ss.shape[0]
			row_label=f'{name}-{id}'
			this_col_labels=[f'{resnum:d}-{ss_names[ss_type]}' for resnum in resnums]
			try:
				ss_frac_df.loc[row_label,this_col_labels]=frac[ires_list]
			except Exception as e:
				print(f'Error in secondary structure assignment for {row_label}: {e}')




active_df=pd.read_csv('relative-population-active-multi-traj.csv')
#rmsd_df=pd.read_csv('egfr-rmsd-sub_trajectory-average.csv')
#rmsd_df.rename(columns={'0':'RMSD'},inplace=True)

#rmsf_df=pd.read_csv('rmsf-by-section.csv')
#rmsf_df.rename(columns={'Unnamed: 0':'Protein'})
#Merge the active, RMSD, and secondary structure dataframes into one dataframe
pdb.set_trace()
#merged_df=pd.merge(active_df,rmsd_df,left_on='Unnamed: 0',right_on='Unnamed: 0',how='inner')
#merged_df=pd.merge(active_df,ss_frac_df,left_on=active_df.index,right_on=ss_frac_df.index)
#merged_df.set_index('Unnamed: 0',inplace=True)
active_df.set_index('Unnamed: 0',inplace=True)
merged_df=pd.merge(active_df,ss_frac_df,left_on=active_df.index,right_on=ss_frac_df.index)
names_from_df=merged_df['key_0']
features_df=merged_df.copy()
features_df.drop(columns='key_0',inplace=True)

#now create an array, one for each mutant/replica, with 0=benign, 1=disease-causing
disease_causing=np.ones((len(names),nreplica))
for iname,name in enumerate(names):
	if (name in benign_mutants):
		disease_causing[iname,:]=0

disease_causing=disease_causing.reshape((len(names)*nreplica,))


#groups=[]
#igroup=1
#the data points are in order by mutation
#for name in names:
#	groups=groups+[igroup]*nsection
#	igroup+=1

#if ss_frac_df is the features dataframe and disease_causing is the target array
features=features_df.to_numpy()
target=disease_causing


#Start by using all of the data and try to estimate the out-of-bag error as a funciton of the number of estimators.
oob_errors=[]
n_estimators_list=list(range(20,100,5))
for n_estimators in n_estimators_list:
	print(f'fitting random forests for {n_estimators:d} estimators',flush=True)
	raf=RandomForestClassifier(n_estimators=n_estimators,max_depth=maxdepth,class_weight='balanced',bootstrap=True,
		oob_score=True,max_features="sqrt",random_state=seed,verbose=1)
	raf.fit(features,target)
	#the error rate is 1 - the OOB score
	oob_errors.append(1-raf.oob_score_)
print(oob_errors,flush=True)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(n_estimators_list,oob_errors)
ax.set_xlabel('number of trees')
ax.set_ylabel('OOB error rate')
ax.set_ylim(bottom=0,top=None)
fig.tight_layout()
fig.savefig('plots/egfr-random-forests-sec_structure-active-oob-error-rate.png',dpi=600)
plt.close(fig)

n_estimators=55 #this was chosen from the plot
print(f'fitting final random forests classifier with {n_estimators:d} trees',flush=True)
#raf=RandomForestClassifier(n_estimators=n_estimators,class_weight='balanced',bootstrap=True,oob_score=True,max_features="sqrt",random_state=seed,verbose=1)
n_splits=4 #this is the number of cross validations
#skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
#skf=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)


#for each estimator, determine the false and true positive rates
fpr=[]
tpr=[]
aucs=[]
feature_importances_all=[]
accuracies_train=[]
accuracies_test=[]
predict_probs_all=np.full(len(features),np.nan,dtype=np.float32) #changed names to features
for irepeat in range(1,nrepeat+1):
	print(f'beginning cross validation repetition {irepeat:d}',flush=True)
	skf=StratifiedKFold(n_splits=n_splits,shuffle=True)
	for fold,(train_idx, test_idx) in enumerate(skf.split(features,target)):
		X_train,X_test=features[train_idx],features[test_idx]
		y_train,y_test=target[train_idx],target[test_idx]
		train_benign=np.count_nonzero(y_train==0)
		train_disease_causing=np.count_nonzero(y_train==1)
		n_train=train_benign+train_disease_causing
		test_benign=np.count_nonzero(y_test==0)
		test_disease_causing=np.count_nonzero(y_test==1)
		n_test=test_benign+test_disease_causing
		print(f'Fold{fold+1} training set has {train_benign:d} benign and {train_disease_causing:d} disease causing samples',flush=True)
		print(f'Fold{fold+1} test set has {test_benign:d} benign and {test_disease_causing:d} disease causing samples',flush=True)

		raf=RandomForestClassifier(n_estimators=n_estimators,max_depth=maxdepth,class_weight='balanced',bootstrap=True,oob_score=True,max_features="sqrt",random_state=seed,verbose=1)
		raf.fit(X_train,y_train)
		print(f'Fold{fold+1} has OOB score {raf.oob_score_:.4f}',flush=True)
		#column zero of predict_probs is the probability of being benign, column 1 that of disease_causing
		predict_probs=raf.predict_proba(X_test)
		predict_probs_all[test_idx]=predict_probs[:,1]
		accuracy_train=accuracy_score(y_true=y_train, y_pred=raf.predict(X_train))
		accuracy_test=accuracy_score(y_true=y_test, y_pred=raf.predict(X_test))
		print(f'Fold{fold+1} has accuracy {accuracy_train:.2f} on training set and accuracy {accuracy_test:.2f} on the test set',flush=True)
		accuracies_train.append(accuracy_train)
		accuracies_test.append(accuracy_test)
		feature_importances_all.append(raf.feature_importances_)

		predict_probs=raf.predict_proba(X_test)
		#disease causing is the "positive" class
		this_fpr, this_tpr, thresholds=roc_curve(y_true=y_test, y_score=predict_probs[:,1], pos_label=1)
		fpr.append(this_fpr)
		tpr.append(this_tpr)
		this_auc=auc(this_fpr,this_tpr)
		print(f'Fold{fold+1} has AUC value {this_auc:.02f}',flush=True)
		aucs.append(auc(this_fpr,this_tpr))


#The AUC is trivially 1.0 for a model trained on all the data.
#predict_probs=raf_all_data.predict_proba(features)
#fpr_all_data, tpr_all_data, thresholds=roc_curve(y_true=target, y_score=predict_probs[:,1], pos_label=1)
#auc_all_data=auc(fpr_all_data,tpr_all_data)
#print(f'AUC score for model trained on all data: {auc_all_data:.2f}',flush=True)

#This calculates the mean ROC curve, mean and sd auc
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []


for idx in range(n_splits):
    interp_tpr = np.interp(mean_fpr, fpr[idx], tpr[idx])
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(np.array(aucs))
sterr_auc = std_auc*np.sqrt(1/(n_splits*nrepeat)+n_test/n_train)
print(f'mean AUC: {mean_auc:.2f} +/- {sterr_auc:.2f}',flush=True)
#Plot without using the RocCurveDisplay, which does not work well with cross validation in sklearn versions less than 1.7.

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#for idx in range(n_splits):
#	ax.plot(fpr[idx],tpr[idx])
ax.plot(
    mean_fpr,
    mean_tpr,
    color="k",
#    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

#ax.plot(fpr_all_data, tpr_all_data, color='red',label=f'All data ROC (AUC={auc_all_data:.2f}',lw=2, alpha=0.8)
std_tpr = np.std(interp_tprs, axis=0)
sterr_tpr = std_tpr*np.sqrt(1/(n_splits*nrepeat)+n_test/n_train)
tprs_upper = np.minimum(mean_tpr + sterr_tpr, 1)
tprs_lower = np.maximum(mean_tpr - sterr_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
#    label=r"$\pm$ 1 std. dev.",
)
ax.set_aspect('equal')
ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
#    title=f"Mean ROC curve with variability",
)
#ax.legend(loc="lower right",fontsize=11)
ax.axline((0,0),(1,1),c='k',ls='--')
plt.tight_layout()
plt.savefig('plots/egfr-random-forests-sec_structure-active-roc-curve.png',dpi=600)
plt.close()

#Save a CSV file with all of the predicted probabilities as well as the correct answer
#pdb.set_trace()
predict_probs_df=pd.DataFrame(predict_probs_all,index=names_from_df) #changed names to section_names

predict_probs_df.insert(len(predict_probs_df.columns),'correct answer',disease_causing)
predict_probs_df.to_csv('egfr-random-forests-sec_structure-active-predict-probs.csv')



#train and characterize a model based on all the data -- it is tested on the same data it is trained on!
raf_all_data=RandomForestClassifier(n_estimators=n_estimators,max_depth=maxdepth,class_weight='balanced',bootstrap=True,oob_score=True,max_features="sqrt",random_state=seed,verbose=1)
raf_all_data.fit(features,target)
oob_score_all_data=raf_all_data.oob_score_
print(f'OOB score for model trained on all data: {oob_score_all_data:.4f}',flush=True)
accuracy_all_data=accuracy_score(y_true=target, y_pred=raf_all_data.predict(features))
print(f'Accuracy for model trained on all data: {accuracy_all_data:.2f}',flush=True)

#display individual trees for the all-data model (this could be put inside the loop if we want the trees from every cross-valiation fold)
for i in range(0,n_estimators):
	print(f'plotting tree {i+1:d}',flush=True)
	clf=raf_all_data.estimators_[i]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	tree.plot_tree(clf,ax=ax,feature_names=features_df.columns)
	outfname=f'trees-sec_structure-active/egfr-random-forests-sec_structure-active-{i+1:d}.png'
	fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
	plt.close(fig)


#Try to plot feature importances.
pdb.set_trace()
feature_importances=np.stack(feature_importances_all)
mean_feature_importances=np.mean(feature_importances,axis=0)
se_feature_importances=np.std(feature_importances,axis=0)/np.sqrt(n_splits)
#they are active, RMSD, "helix", "sheet", "helix", "sheet", etc.
mean_feature_importances_active=mean_feature_importances[0]
se_feature_importances_active=se_feature_importances[0]
#mean_feature_importances_rmsd=mean_feature_importances[1]
#se_feature_importances_rmsd=se_feature_importances[1]
print(f'feature importance for fraction active: {mean_feature_importances_active:.4f} +/- {se_feature_importances_active:.4f}',flush=True)
#print(f'feature importance for RMSD: {mean_feature_importances_rmsd:.4f} +/- {se_feature_importances_rmsd:.4f}',flush=True)

mean_feature_importances_helix=mean_feature_importances[1::2]
se_feature_importances_helix=se_feature_importances[1::2]
mean_feature_importances_sheet=mean_feature_importances[2::2] 
se_feature_importances_sheet=se_feature_importances[2::2]

#plot vs. residue number -- resnums contains the residue numbers
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(resnums,mean_feature_importances_helix,color='C0',label='helix')
ax.fill_between(resnums,mean_feature_importances_helix-se_feature_importances_helix,mean_feature_importances_helix+se_feature_importances_helix,
	color='C0',alpha=0.2)
ax.plot(resnums,mean_feature_importances_sheet,color='C1',label='sheet')
ax.fill_between(resnums,mean_feature_importances_sheet-se_feature_importances_sheet,mean_feature_importances_sheet+se_feature_importances_sheet,
        color='C1',alpha=0.2)
ax.set_xlabel('residue number')
ax.set_ylabel('feature importance')
ax.legend(loc='upper right')
xticks=np.array(list(range(700,901,100))+[985])
ax.set_xticks(xticks)
#ax.set_xticklabels(xticks)
xticks=np.array([696]+list(range(700,981,20))+[985])
ax.set_xticks(xticks,minor=True)
plt.tight_layout()
plt.savefig('plots/egfr-random-forests-sec_structure-active-importance.png',dpi=600)
plt.close()

#dump the feature importances to a file
#output=open('egfr-random-forests-sec_structure-active

sys.exit(0)
