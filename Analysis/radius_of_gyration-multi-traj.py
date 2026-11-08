#!/usr/bin/env python

import sys
import math
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pandas as pd
import pdb

mut_type=sys.argv[1]
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

#pdb.set_trace()

rg_data_list=[]

name_ids=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
mean_rg_df=pd.DataFrame(index=name_ids,columns=['mean RG'])

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ideally we would determine this from the psfs, or pass it as a parameter, but we will hard-code it here for now 
#res_start=696
#res_end=985
#nres=res_end-res_start+1
#this creates a 2D array with the first axis being residue number (from 0 to nres-1!) and the second being the mutants
#the data type of the array is single-precision floating point numbers, and it is initially filled with NaNs. (Not a Number)
#hard-code the length of the trajectory for now
#nframes=1000000//stride
#rmsd_data=np.full((len(names),nframes),np.nan,dtype=np.float32)

#block_sizes=[1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000]

outfname=f'egfr-rg-{mut_type}'
output=open(outfname,'w')
for imut,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	reffname='data/{0}-solvated.pdb'.format(name)
	reftraj=md.load(reffname)
	mean_rg_all=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#pdb.set_trace()
		#compute radius of gyration over the whole trajectory
		rg=md.compute_rg(traj)*10 #nm to A
		rg_data_list.append(rg)
		mean_rg_all.append(np.mean(rg))
		mean_rg_df.at[f'{name}-{id}','mean RG']=np.mean(rg)


	#calculate overall mean and SE of the mean RMSD from all the frames
	mean_rg_all=np.array(mean_rg_all)
	overall_mean_rg_all=np.mean(mean_rg_all)
	overall_se_rg_all=np.std(mean_rg_all,ddof=1)/np.sqrt(len(mean_rg_all))
	output.write(f'{name} {overall_mean_rg_all:.2f} +/- {overall_se_rg_all:.2f}\n')

output.close()

#save the RMSD data for each trajectory to csv file for use with random forests
mean_rg_df.to_csv('egfr-radius-of-gyration.csv')

#plot the time course radius of gyration
#pad the rg_data_list to have uniform length
max_frames=max(len(rg) for rg in rg_data_list)
rg_data=np.full((len(rg_data_list),max_frames),np.nan,dtype=np.float32)
for i,rg in enumerate(rg_data_list):
	rg_data[i,:len(rg)]=rg

#the vmin and vmax give the colorbar range
_=ax.matshow(rg_data,cmap='Greys',vmin=19.5,vmax=20.5)
cb=fig.colorbar(_,shrink=0.8,aspect=10)
cb.set_label(label='Radius of Gyration ($\\AA$)',fontsize=11)
#finish and save the plot
ax.set_xlim(left=0,right=1000)
ax.set_xlabel('time (ns)')
ax.set_aspect(30)
if (mut_type=='pathogenic'):
	ax.set_aspect(15)
#this should ensure that the labels are in the same order as the names and in the correct positions
yticklabels=[]
for name in names:
	for id in range(1,nreplica+1):
		yticklabels.append(f'{labels[name]} rep. {id:d}')
ax.set_ylabel('Radius of Gyration ($\\AA$)')
ax.set_yticks(np.arange(len(rg_data)),yticklabels)
if (mut_type=='benign'):
	ax.tick_params(axis='y',labelsize=11)
elif (mut_type=='pathogenic'):
	ax.tick_params(axis='y',labelsize=5)
#ax.set_yticklabels(yticklabels)
outfname=f'plots/egfr-rg-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)
