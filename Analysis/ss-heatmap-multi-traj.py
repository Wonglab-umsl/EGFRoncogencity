#!/usr/bin/env python

import sys
import math
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pickle
import pdb

ss_types={'helix':'H','sheet':'E'}

mut_type=sys.argv[1]
nreplica=int(sys.argv[2])
ss_name=sys.argv[3] #either 'helix' or 'sheet' 
ss_type=ss_types[ss_name]
stride=1
if (len(sys.argv)>4):
        stride=int(sys.argv[4])
#time between frames in ns
dt=0.001*stride 
#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
#scratch='/cluster/pixstor/wongch-lab/Victor/egfr//dcd-joined/'
#we will add the names of the mutants later

input=open('/cluster/pixstor/wongch-lab/Victor/egfr/egfr-sec-struc-info.p','rb')
ss=pickle.load(input)
input.close()

#pdb.set_trace()


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

#rmsd_data_list=[]

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


# Find the maximum number of frames across all trajectories
max_frames = max([ss_info.shape[0] for ss_info in ss.values()])
# Create a 2D array with NaN padding for RMSD data
ss_frac=np.full((len(names)*nreplica, max_frames), np.nan, dtype=np.float32) 
#here imut actually indexes over mutation and replica
for iname, name in enumerate(names):
	for id in range(1,nreplica+1):
		idx=iname*nreplica+id-1
		ss_info=ss[(name,id)][::stride]
		nres=ss_info.shape[1]
		ss_frac[idx, :ss_info.shape[0]] = np.count_nonzero(ss_info==ss_type,axis=1)/nres


#print('RMSD range {0:.3f} - {1:.3f}'.format(np.min(rmsd_data),np.max(rmsd_data)),flush=True)
#the vmin and vmax give the colorbar range
if (ss_name=='helix'):
	vmin=0.3
	vmax=0.45
elif (ss_name=='sheet'):
	vmin=0.1
	vmax=0.25

_=ax.matshow(ss_frac,cmap='viridis',vmin=vmin,vmax=vmax)
cb=fig.colorbar(_,shrink=0.8,aspect=10)
cb.set_label(label=f'frac. {ss_name}',fontsize=11)
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
ax.set_ylabel(f'frac. {ss_name} sec. structure')
ax.set_yticks(np.arange(len(names)*nreplica),yticklabels)
if (mut_type=='benign'):
	ax.tick_params(axis='y',labelsize=11)
elif (mut_type=='pathogenic'):
	ax.tick_params(axis='y',labelsize=6)
#ax.set_yticklabels(yticklabels)
outfname=f'plots/egfr-ss-heatmap-{mut_type}-{ss_name}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)

