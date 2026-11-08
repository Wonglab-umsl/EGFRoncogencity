#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import os
import pdb
import pickle
import pandas as pd
from sklearn import tree
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, accuracy_score


seed=None
#number of dimensions and bin size for histograms
nreplica=int(sys.argv[1])
maxdepth=int(sys.argv[2])
nrepeat=int(sys.argv[3])
ncomponents=int(sys.argv[4])
binsize=float(sys.argv[5])

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

os.makedirs('trees-pca', exist_ok=True)
os.makedirs('plots', exist_ok=True)

benign_mutants=['egfr-wt-active-charmm','egfr-v904i-active-charmm','egfr-i926n-active-charmm','egfr-i966v-active-charmm','egfr-f712l-active-charmm','egfr-s720f-active-charmm','egfr-p741s-active-charmm','egfr-i744m-active-charmm','egfr-v774l-active-charmm','egfr-r831h-active-charmm']

#unpickle the file
input=open('egfr-pca-distances-reduced-all-data2.p','rb')
frame_start=pickle.load(input)
frame_end=pickle.load(input)
pca_coords_all=pickle.load(input)
input.close()
#pdb.set_trace()

#now create an array, one for each mutant/replica, with 0=benign, 1=disease-causing
#disease_causing=np.ones((len(names),nreplica))
#for i,sec_name in enumerate(traj_names):
#for iname in enumerate(names):
#	if (name in benign_mutants):
#		disease_causing[iname,:]=0
#disease_causing=disease_causing.reshape((len(names)*nreplica,))

disease_causing=np.ones((len(names),nreplica))
for iname,name in enumerate(names):
        if (name in benign_mutants):
                disease_causing[iname,:]=0

disease_causing=disease_causing.reshape((len(names)*nreplica,))

#create a histogram for each trajector
#here we find the range of all the data in both dimensions, then set up grid boundaries
binmin=np.floor(np.min(pca_coords_all,axis=0)/binsize)*binsize
binmax=np.ceil(np.max(pca_coords_all,axis=0)/binsize)*binsize
nbin=((binmax-binmin)/binsize).astype(int)
grid_bins=[np.linspace(start=binmin[icomp], stop=binmax[icomp], num=nbin[icomp] + 1) for icomp in range(0,ncomponents)]
histograms=[]
for name in names:
	for id in range(1,nreplica+1):
		start=frame_start[(name,id)]
		end=frame_end[(name,id)]
		H,edges=np.histogramdd(pca_coords_all[start:end+1,0:ncomponents],bins=grid_bins,density=True)
		# Normalize density
		H_normalized=H/H.sum()
		#flatten the histogram so that we ultimately have a 2D array with name on the first dimension, histogram on the seond
		histograms.append(H_normalized.flatten())
histograms=np.stack(histograms,axis=0)

#interlude: compare mean and SE of benign and pathogenic histograms to see where there are systematic differences
#compute mean and SE of the grid densities for each important feature for benign/pahtogenic
benign_idx=np.where(disease_causing==0)[0]
pathogenic_idx=np.where(disease_causing==1)[0]
mean_histograms_benign=np.mean(histograms[benign_idx,:],axis=0)
se_histograms_benign=np.std(histograms[benign_idx,:],axis=0,ddof=1)/np.sqrt(len(benign_idx))
mean_histograms_pathogenic=np.mean(histograms[pathogenic_idx,:],axis=0)
se_histograms_pathogenic=np.std(histograms[pathogenic_idx,:],axis=0,ddof=1)/np.sqrt(len(pathogenic_idx))
#this is the t-score for two means assuming unequal variances
t_scores=(mean_histograms_pathogenic-mean_histograms_benign)/np.sqrt(se_histograms_benign**2+se_histograms_pathogenic**2)
#it will have nan's where the standard errors are both zero and we cannot make a comparison
t_scores=np.nan_to_num(t_scores,nan=0.0)
if (ncomponents==2):
	vmax=np.max(np.abs(t_scores))
	vmin=-vmax
	t_scores=t_scores.reshape(nbin[0:ncomponents])
	fig=plt.figure(figsize=(8, 6))
	_=plt.imshow(t_scores, origin='lower', aspect='auto', extent=[binmin[0],binmax[0],binmin[1],binmax[1]], cmap='RdBu', vmin=vmin, vmax=vmax)
	plt.colorbar(label='t-score for density difference')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.tight_layout()
	plt.savefig('plots/egfr-pca-density-difference.png',dpi=600)
	plt.close()





#add the column to the merged data set
#merged_df.insert(loc=1,column='disease causing',value=disease_causing)
#features_train,features_test,dc_train,dc_test=train_test_split(features_df,disease_causing,train_size=0.5,shuffle=True,stratify=disease_causing)
#features_train,dc_train,features_test,dc_test=equal_sampler(classes=np.array([0,1]),data=histograms,target=disease_causing,test_frac=0.5)
#we'll just label the columns with a histogram box number
column_labels=list(range(1,histograms.shape[1]+1))


#if ss_frac_df is the features dataframe and disease_causing is the target array
features=histograms
target=disease_causing

print("Features shape:", features.shape)
print("Target shape:", target.shape)


#Start by using all of the data and try to estimate the out-of-bag error as a funciton of the number of estimators.
oob_errors=[]
n_estimators_list=list(range(20,100,5))
for n_estimators in n_estimators_list:
	print(f'fitting random forests for {n_estimators:d} estimators',flush=True)
	raf=RandomForestClassifier(n_estimators=n_estimators,class_weight='balanced',max_depth=maxdepth,bootstrap=True,oob_score=True,max_features="sqrt",random_state=123,verbose=1)
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
fig.savefig('plots/egfr-random-forests-pca-oob-error-rate.png',dpi=600)
plt.close(fig)

n_estimators=100 #this was chosen from the plot
print(f'fitting final random forests classifier with {n_estimators:d} trees',flush=True)
raf=RandomForestClassifier(n_estimators=n_estimators,class_weight='balanced',max_depth=maxdepth,bootstrap=True,oob_score=True,max_features="sqrt",random_state=123,verbose=1)
n_splits=4 #this is the number of cross validations


#groups for ensuring that each mutant is either in the training set or test set
pdb.set_trace()
groups=np.concatenate([np.repeat(i,nreplica) for i in range(1,len(names)+1)])


#for each estimator, determine the false and true positive rates
fpr=[]
tpr=[]
aucs=[]
feature_importances_all=[]
#This is a 2D table by data item and by cross-validation folds)
#predict_probs_all=np.full((len(names),n_splits),np.nan,dtype=np.float32)
#this is 1D, doesn't keep track of the folds (each data item is in one of the cross-validation folds)
predict_probs_all=np.full(len(names)*nreplica,np.nan,dtype=np.float32)
for irepeat in range(1,nrepeat+1):
	print(f'beginning cross validation repetition {irepeat:d}',flush=True)
	skf=StratifiedGroupKFold(n_splits=n_splits,shuffle=True)
	for fold,(train_idx, test_idx) in enumerate(skf.split(features,target,groups=groups)):
		X_train,X_test=features[train_idx],features[test_idx]
		y_train,y_test=target[train_idx],target[test_idx]
		train_benign=np.count_nonzero(y_train==0)
		train_disease_causing=np.count_nonzero(y_train==1)
		n_train=train_benign+train_disease_causing
		test_benign=np.count_nonzero(y_test==0)
		test_disease_causing=np.count_nonzero(y_test==1)
		n_test=test_benign+test_disease_causing
		print(f'Fold{fold+1} training set has {train_benign:d} benign and {train_disease_causing:d} disease causing samples, total {n_train:d} samples',flush=True)
		print(f'Fold{fold+1} test set has {test_benign:d} benign and {test_disease_causing:d} disease causing samples, total {n_test:d} samples',flush=True)


		raf=RandomForestClassifier(n_estimators=n_estimators,class_weight='balanced',max_depth=maxdepth,bootstrap=True,oob_score=True,max_features="sqrt",verbose=1)
		raf.fit(X_train,y_train)
		predict_probs=raf.predict_proba(X_test)
		#column zero of predict_probs is the probability of being benign, column 1 that of disease_causing)
		predict_probs_all[test_idx]=predict_probs[:,1]
		feature_importances_all.append(raf.feature_importances_)
		#disease causing is the "positive" class
		#pdb.set_trace()
		this_fpr, this_tpr, thresholds=roc_curve(y_true=y_test, y_score=predict_probs[:,0], pos_label=0)
		fpr.append(this_fpr)
		tpr.append(this_tpr)
		this_auc=auc(this_fpr,this_tpr)
		print(f'Fold{fold+1} has AUC value {this_auc:.02f}',flush=True)
		aucs.append(auc(this_fpr,this_tpr))


#This calculates the mean ROC curve, mean and sd auc
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []


#for idx in range(n_splits):
for idx in range(len(aucs)):
    interp_tpr = np.interp(mean_fpr, fpr[idx], tpr[idx])
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(np.array(aucs))
sterr_auc = std_auc*np.sqrt(1/(len(aucs))+1)
#sterr_auc = std_auc*np.sqrt(1/(n_splits*nrepeat)+n_test/n_train)
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

std_tpr = np.std(interp_tprs, axis=0)
sterr_tpr = std_tpr*np.sqrt(1/(len(aucs))+1)
#sterr_tpr = std_tpr*np.sqrt(1/(n_splits*nrepeat)+n_test/n_train)
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
#    title=f"Mean ROC curve with variability')",
)
#ax.legend(loc="lower right",fontsize=11)
ax.axline((0,0),(1,1),c='k',ls='--')
plt.tight_layout()
plt.savefig('plots/egfr-random-forests-pca-roc-curve.png',dpi=600)
plt.close()


#Save a CSV file with all of the predicted probabilities as well as the correct answer
#pdb.set_trace()
name_ids=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
predict_probs_df=pd.DataFrame(predict_probs_all,index=name_ids) #changed names to section_names
predict_probs_df.insert(len(predict_probs_df.columns),'correct answer',disease_causing)
predict_probs_df.to_csv('egfr-random-forests-pca-predict-probs.csv')



#train and characterize a model based on all the data -- it is tested on the same data it is trained on!
raf_all_data=RandomForestClassifier(n_estimators=n_estimators,max_depth=maxdepth,class_weight='balanced',bootstrap=True,oob_score=True,max_features="sqrt",random_state=seed,verbose=1)
raf_all_data.fit(features,target)
oob_score_all_data=raf_all_data.oob_score_
print(f'OOB score for model trained on all data: {oob_score_all_data:.4f}',flush=True)
accuracy_all_data=accuracy_score(y_true=target, y_pred=raf_all_data.predict(features))
print(f'Accuracy for model trained on all data: {accuracy_all_data:.2f}',flush=True)

#display individual trees for the all-data model (this could be put inside the loop if we want the trees from every cross-valiation fold)
plot_trees=False
if (plot_trees):
	for i in range(0,n_estimators):
		print(f'plotting tree {i+1:d}',flush=True)
		clf=raf_all_data.estimators_[i]
		fig=plt.figure()
		ax=fig.add_subplot(1,1,1)
		tree.plot_tree(clf,ax=ax,feature_names=[str(i) for i in column_labels])
		#tree.plot_tree(clf,ax=ax,feature_names=features_df.columns)
		outfname=f'trees-pca/egfr-random-forests-pca-{i+1:d}.png'
		fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
		plt.close(fig)


#Try to plot feature importances.
feature_importances=np.stack(feature_importances_all)
mean_feature_importances=np.mean(feature_importances,axis=0)
se_feature_importances=np.std(feature_importances,axis=0,ddof=1)/np.sqrt(n_splits*nrepeat)

if (ncomponents==2):
	new_mean_feature_importances=mean_feature_importances.reshape(nbin[0:ncomponents])
	plt.figure(figsize=(8, 6))
	plt.imshow(new_mean_feature_importances, origin='lower', aspect='auto', extent=[binmin[0],binmax[0],binmin[1],binmax[1]])
	plt.colorbar(label='Feature Importance')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.tight_layout()
	plt.savefig('plots/egfr-random-forests-pca-importance.png',dpi=600)
	plt.close()





# Get feature (column) names
#feature_names=features_df.columns.to_numpy()
#feature_names=np.array(column_labels)

# Sort features by importance
sorted_idx=np.argsort(mean_feature_importances)[::-1]
sorted_importances=mean_feature_importances[sorted_idx]
#This is the grid density by mutant for the top 10 most important grid boxes.
n_important_features=20
important_features=histograms[:,sorted_idx[0:n_important_features]]
#split this by benign and pathogenic
#benign_idx=np.where(disease_causing==0)[0]
important_features_benign=important_features[benign_idx,:]
#pathogenic_idx=np.where(disease_causing==1)[0]
important_features_pathogenic=important_features[pathogenic_idx,:]

#this reorders everything so that benign is on top and pathogenic on the bottom
important_features_reordered=np.vstack([important_features_benign,important_features_pathogenic])
#normalize rel. to max grid density for particular feature
important_features_reordered=important_features_reordered/np.max(important_features,axis=0)




#reorder the labels
#pdb.set_trace()
name_ids_reordered=[name_ids[i] for i in benign_idx] + [name_ids[i] for i in pathogenic_idx]

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
_=ax.imshow(important_features_reordered,cmap='viridis')
cb=fig.colorbar(_)
cb.set_label('normalized grid density')
ax.set_xticks(range(0,n_important_features))
ax.set_xticklabels(range(1,n_important_features+1))
ax.set_yticks(range(0,len(names)*nreplica))
ax.set_yticklabels(name_ids_reordered)
benign_pos=len(benign_idx)
ax.axhline(benign_pos,color='white')
ax.set_aspect(0.2)
ax.tick_params(axis='x',labelsize=11)
ax.tick_params(axis='y',labelsize=5)
fig.savefig('plots/egfr-random-forests-pca-density-important-features.png',format='png',bbox_inches='tight',dpi=600)
plt.close(fig)



#sorted_errors=se_feature_importances[sorted_idx]
#sorted_names=feature_names[sorted_idx]



#plot vs. residue number -- resnums contains the residue numbers
#fig=plt.figure(figsize=(10,6))
#ax=fig.add_subplot(1,1,1)
#ax.bar(range(len(sorted_importances)),sorted_importances,yerr=sorted_errors,align='center',color='C0',alpha=0.7)
#ax.set_xticks(range(len(sorted_names)))
#ax.set_xticklabels(sorted_names,rotation=90,fontsize=8)
#ax.set_ylabel('Feature Importance')
#ax.set_title('Random Forest Feature Importances')
#plt.tight_layout()
#plt.savefig('plots/egfr-random-forests-pca-importance.png',dpi=600)
#plt.close()


sys.exit(0)
