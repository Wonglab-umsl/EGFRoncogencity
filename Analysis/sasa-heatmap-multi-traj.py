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

sasa_data_list=[]

name_ids=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
mean_sasa_df=pd.DataFrame(index=name_ids,columns=['mean SASA'])

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


#SASA is computed over the whole trajectories
outfname=f'egfr-sasa-{mut_type}'
output=open(outfname,'w')
for imut,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	reffname='data/{0}-solvated.pdb'.format(name)
	reftraj=md.load(reffname)
	mean_sasa_all=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#compute SASA using Shrake-Rupley algorithm
		#returns arrays of shape (n_frames, n_atoms) with SASA in A^2
		sasa=md.shrake_rupley(traj)

		#sum SASA over all atoms to get a total SASA per frame-- convert nm^2 to A^2
		sasa_total=np.sum(sasa,axis=1)*100
		#pdb.set_trace()
		sasa_data_list.append(sasa_total)
		mean_sasa_all.append(np.mean(sasa_total))
		mean_sasa_df.at[f'{name}-{id}','mean SASA']=np.mean(sasa_total)

	#calculate overall mean and SE of the mean SASA from all frames of each trajectory
	mean_sasa_all=np.array(mean_sasa_all)
	overall_mean_sasa_all=np.mean(mean_sasa_all)
	overall_se_sasa_all=np.std(mean_sasa_all,ddof=1)/np.sqrt(len(mean_sasa_all))
	output.write(f'{name} {overall_mean_sasa_all:.2f} +/- {overall_se_sasa_all:.2f}\n')

output.close()

#save the SASA data for each trajectory to csv file for use with random forests
mean_sasa_df.to_csv('egfr-mean-sasa.csv')

# Find the maximum number of frames across all trajectories
max_frames = max([len(sasa) for sasa in sasa_data_list])

# Create a 2D array with NaN padding for SASA data
sasa_data=np.full((len(sasa_data_list), max_frames), np.nan, dtype=np.float32)
for imut, sasa in enumerate(sasa_data_list):
	sasa_data[imut, :len(sasa)] = sasa


print('SASA range {0:.3f} - {1:.3f}'.format(np.min(sasa_data),np.max(sasa_data)),flush=True)
#the vmin and vmax give the colorbar range
_=ax.imshow(sasa_data,cmap='viridis') #,vmin=0,vmax=3.5)
cb=fig.colorbar(_,shrink=0.8,aspect=10)
cb.set_label(label='Total SASA ($\\AA^2$)',fontsize=11)
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
ax.set_ylabel('Total SASA ($\\AA^2$)')
ax.set_yticks(np.arange(len(sasa_data_list)),yticklabels)
if (mut_type=='benign'):
	ax.tick_params(axis='y',labelsize=10)
elif (mut_type=='pathogenic'):
	ax.set_aspect(6)
#if (mut_type=='benign'):
#	ax.set_aspect(10)
#elif (mut_type=='pathogenic'):
#	ax.set_aspect(5)
#ax.set_yticklabels(yticklabels)
outfname='plots/egfr-sasa-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)


