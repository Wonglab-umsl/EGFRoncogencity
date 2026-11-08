#!/usr/bin/env python

import sys
import math
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pdb
import pandas as pd

mut_type=sys.argv[1] #either "benign" or "pathogenic"
nreplica=int(sys.argv[2])
stride=1
if (len(sys.argv)>3):
        stride=int(sys.argv[3])
#time between frames in ns
dt=0.001*stride 
#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
scratch='/cluster/pixstor/wongch-lab/Victor/egfr//dcd-joined/'
#we will add the names of the mutants later

input=open(f'list-{mut_type}','r')
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


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ideally we would determine this from the psfs, or pass it as a parameter, but we will hard-code it here for now 
res_start=696
res_end=985
nres=res_end-res_start+1
#this creates a 2D array with the first axis being residue number (from 0 to nres-1!) and the second being the mutants
#the data type of the array is single-precision floating point numbers, and it is initially filled with NaNs. (Not a Number)
rmsf_data=np.full((len(names),nres),np.nan,dtype=np.float32)
rmsf_se_data=np.full((len(names),nres),np.nan,dtype=np.float32)
for imut,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	reffname='data/{0}-solvated.pdb'.format(name)
	reftraj=md.load(reffname)
	rmsf_all=[] #this is a list of all the RMSFs for each replica
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#RMSD align trajectory against reference
		backbone_atoms=top.select('(name N or name CA or name C) and (residue 705 to 974)')
		traj.superpose(reftraj,atom_indices=backbone_atoms)	

		#compute RMSFs -- call md.rmsf
		ca_atoms=top.select('name CA')
		#I think this uses the average position on the trajectory 
		rmsf=md.rmsf( traj, traj, atom_indices=ca_atoms )*10
		rmsf_all.append(rmsf)
		del traj
	rmsf_all=np.stack(rmsf_all)
	mean_rmsf=np.mean(rmsf_all,axis=0)
	se_rmsf=np.std(rmsf_all,axis=0,ddof=1)/np.sqrt(nreplica)
	#add some python code here to take values from the rmsf array and put them in the correct places in the rmsf_data array.
	#hint: top.atom(idx).residue.resSeq gives the residue number for atom idx
	#rmsf is indexed by atoms within ca_atoms
	for idx, atom_idx in enumerate(ca_atoms):
		res=top.atom(atom_idx).residue.resSeq
		if ((res==int(res)) and (res_start <= res) and (res <= res_end)):
			rmsf_data[imut, res - res_start] = mean_rmsf[idx]
			rmsf_se_data[imut, res - res_start] = se_rmsf[idx]
	#write the same data out to a file so we can look at it numerically
	#outfname='rmsf-{0}'.format(name)
	#output=open(outfname,'w')
	#for i,res in enumerate(resnums):
	#	line='{0:d} {1:.3f}\n'.format(res,rmsf[i])
	#	output.write(line)
	#output.close()

print('RMSF range {0:.3f} - {1:.3f} A'.format(np.nanmin(rmsf_data),np.nanmax(rmsf_data)),flush=True)
#add an extra row for the max std error over all mutants
rmsf_max_se=np.max(rmsf_se_data,axis=0)
rmsf_data_modified=np.vstack([rmsf_data,rmsf_max_se])
#the vmin and vmax give the colorbar range
_=ax.imshow(rmsf_data_modified,cmap='Greys',vmin=0,vmax=6)
cb=fig.colorbar(_,shrink=0.6,aspect=10)
cb.set_label(label='C$\\alpha$ RMSF ($\\AA$)',fontsize=11)
#finish and save the plot
ax.set_xlim(left=0,right=nres)
ax.set_xlabel('residue number')  #what should it be?
xticks=np.array(list(range(700,901,100))+[985])
ax.set_xticks(xticks-res_start,xticks)
#ax.set_xticklabels(xticks)
xticks=np.array([696]+list(range(700,981,20))+[985])
ax.set_xticks(xticks-res_start,minor=True)
ax.set_aspect(10)
#this should ensure that the labels are in the same order as the names and in the correct positions
yticklabels=[]
for name in names:
	yticklabels.append(labels[name])
yticklabels.append('Max. SE') #for the last row 
ax.set_ylabel('mutant')
ax.set_yticks(np.arange(len(names)+1),yticklabels)
ax.tick_params(axis='y',labelsize=11)
#ax.set_yticklabels(yticklabels)
outfname=f'plots/egfr-rmsf-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)

#pdb.set_trace()
reslist=range(res_start,res_end+1)
rmsd_df=pd.DataFrame(data=rmsf_data,index=names,columns=reslist)
rmsd_df.to_csv('rmsf.csv')

