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

rmsd_data_list=[]

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
for imut,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	reffname='data/{0}-solvated.pdb'.format(name)
	reftraj=md.load(reffname)
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#RMSD align trajectory against reference
		backbone_atoms=top.select('(name N or name CA or name C) and (residue 705 to 974)')
		traj.superpose(reftraj,atom_indices=backbone_atoms)
	
		#compute RMSDs -- convert nm to A
		rmsd=md.rmsd(traj,reftraj,atom_indices=backbone_atoms)*10
		#rmsd_data[imut,:]=rmsd
		rmsd_data_list.append(rmsd)


# Find the maximum number of frames across all trajectories
max_frames = max([len(rmsd) for rmsd in rmsd_data_list])

# Create a 2D array with NaN padding for RMSD data
rmsd_data=np.zeros((len(rmsd_data_list), max_frames), dtype=np.float32) 
#here imut actually indexes over mutation and replica
for imut, rmsd in enumerate(rmsd_data_list):
	rmsd_data[imut, :len(rmsd)] = rmsd


print('RMSD range {0:.3f} - {1:.3f}'.format(np.min(rmsd_data),np.max(rmsd_data)),flush=True)
#the vmin and vmax give the colorbar range
_=ax.matshow(rmsd_data,cmap='Greys',vmin=0,vmax=5)
cb=fig.colorbar(_,shrink=0.8,aspect=10)
cb.set_label(label='C$\\alpha$ RMSD ($\\AA$)',fontsize=11)
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
ax.set_ylabel('backbone RMSD ($\\AA$)')
ax.set_yticks(np.arange(len(rmsd_data_list)),yticklabels)
if (mut_type=='benign'):
	ax.tick_params(axis='y',labelsize=11)
elif (mut_type=='pathogenic'):
	ax.tick_params(axis='y',labelsize=5)
#ax.set_yticklabels(yticklabels)
outfname=f'plots/egfr-rmsd-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)

