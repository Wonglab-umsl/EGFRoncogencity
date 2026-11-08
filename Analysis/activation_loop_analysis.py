#!/usr/bin/env python

import sys
import math
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb
from itertools import combinations
from scipy.spatial.distance import squareform
from datetime import datetime as dtt
import pickle
scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined'

def block_average_se(data,Num_blocks_list=[3,6]):
	se_values=[]
	data=np.asarray(data)
	nframes=len(data)
	for num_blocks in Num_blocks_list:
		block_size=nframes//num_blocks
		if (block_size<2):
			continue
		block_means=[]
		for iblock in range(num_blocks):
			start=iblock*block_size
			if (iblock==num_blocks-1):
				end=nframes
			else:
				end=(iblock+1)*block_size
			block=np.mean(data[start:end])
			block_means.append(block)
		block_means=np.array(block_means)
		if (len(block_means)<2):
			continue
		se=np.std(block_means,ddof=1)/np.sqrt(len(block_means))
		se_values.append(se)
	if (len(se_values)==0):
		return np.nan

	#conservative estimate
	return np.max(se_values)




dt=0.001 #in ns
mut_type=sys.argv[1]
nreplica=int(sys.argv[2])
#res_start=int(sys.argv[3])
#res_end=int(sys.argv[4])


#nres=res_end-res_start+1
stride=1
if (len(sys.argv)>3):
        stride=int(sys.argv[3])


activation_loop_pairs=[(845,871),(847,871),(850,871),(855,871),(858,871)]

open_threshold=15.0


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

fig=plt.figure(constrained_layout=True,figsize=(12,len(names)*nreplica*1.5))

summary=[]
heatmap_data=[]
heatmap_labels=[]

for iname,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	mutant_distances=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)
		#phipsi_this_traj=np.full((traj.n_frames,2*nres),np.nan,dtype=np.float32)
		#Build pairs
		pair_indices=[]
		valid_pairs=[]
		for res1,res2 in activation_loop_pairs:
			sel1 = top.select(f'resSeq == {res1} and name CA')
			sel2 = top.select(f'resSeq == {res2} and name CA')
			print(f'Pair {res1}-{res2}: {len(sel1)} atoms, {len(sel2)} atoms',flush=True)
			if (len(sel1)==0 or len (sel2)==0):
				print(f'warning: could not find pair {res1}-{res2}',flush=True)
				continue
			pair_indices.append((sel1[0], sel2[0]))
			valid_pairs.append((res1, res2))
		if (len(pair_indices)==0):
			print(f'no valid residue pairs for {name}-{id}',flush=True)
			continue

		#Distances
		distances=md.compute_distances(traj,pair_indices)
		distances*=10.0
		activation_loop_distance=np.mean(distances,axis=1)


		#Per-pair statistics
		print("\nPer-pair statistics:")
		for i, (a, b) in enumerate(pair_indices):
			mean_d = np.mean(distances[:, i])
			min_d = np.min(distances[:, i])
			max_d = np.max(distances[:, i])
			print(f"{i} {a}-{b}: mean={mean_d:.2f} Å, min={min_d:.2f}, max={max_d:.2f}")

		#Occupancy
		occupancy=np.sum(activation_loop_distance > open_threshold)/len(activation_loop_distance) 
		mean_distance=np.mean(activation_loop_distance) 
		se_distance=block_average_se(activation_loop_distance,Num_blocks_list=[3,6])
		mutant_distances.append(mean_distance) 
		summary.append([name,id,mean_distance,se_distance,occupancy]) 
		print(f'{name}-{id}: mean distance = {mean_distance:.3f} A',flush=True) 
		print(f'{name}-{id}: block SE = {se_distance:.3f} A',flush=True) 
		print(f'{name}-{id}: occupancy = {occupancy:.3f}',flush=True)

		#Time axis
		frames=np.arange(traj.n_frames)
		time_ns=frames*dt*stride
	
		#plot
		subplot_index=1+(iname*nreplica+id-1)
		ax=fig.add_subplot(len(names)*nreplica,1,subplot_index)
		ax.plot(time_ns,activation_loop_distance,alpha=0.9)
		ax.axhline(y=open_threshold,color='red',linestyle='--',linewidth=1,alpha=0.8)
		ax.set_xlim(left=0,right=time_ns[-1])
		ax.set_ylabel(f'{labels[name]} rep. {id:d}',fontsize=6)
		ax.tick_params(labelsize=8)
		ax.grid(alpha=0.3)
		if (iname==len(names)-1):
			ax.set_xlabel('time (ns)',fontsize=8)
		if (iname==0):
			ax.set_title('Activation-Loop Opening Distance Time Course',fontdict={'fontsize':12})

	#Heatmap
	if (len(mutant_distances)>0):
		heatmap_data.append(mutant_distances)
		heatmap_labels.append(labels[name])

#save plot
fig.subplots_adjust(wspace=0,hspace=0.3)
outfname=f'plots/egfr-activation-loop-distance-{mut_type}-bytime.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)

#write output to file
output=open(f'activation-loop-summary-{mut_type}.dat','w')
output.write('#system replica mean_distance(A) block_SE(A) occupancy\n')

for row in summary:
	output.write(f'{row[0]} {row[1]} {row[2]:.4f} {row[3]:.4f} {row[4]:.4f}\n')
output.close()

#Mutant summary
mutant_summary={}
for row in summary:
	mutant=row[0]
	mean_dist=row[2]
	if mutant not in mutant_summary:
		mutant_summary[mutant]=[]
	mutant_summary[mutant].append(mean_dist)
output=open(f'activation-loop-mutant-summary-{mut_type}.dat','w')

for mutant in mutant_summary.keys():
	values=np.array(mutant_summary[mutant])
	mutant_mean=np.mean(values)
	if (len(values)>1):
		mutant_se=np.std(values,ddof=1)/np.sqrt(len(values))
	else:
		mutant_se=np.nan
	output.write(f'{mutant} {mutant_mean:.4f} {mutant_se:.4f}\n')
output.close()





#Heatmap plot
heatmap_data=np.array(heatmap_data)
fig=plt.figure(figsize=(8,10))
ax=fig.add_subplot(111)
im=ax.imshow(heatmap_data,aspect='auto')
ax.set_yticks(np.arange(len(heatmap_labels)))
ax.set_yticklabels(heatmap_labels)
ax.set_xticks(np.arange(nreplica))
ax.set_xticklabels([f'Rep {i+1}' for i in range(nreplica)])
cbar=plt.colorbar(im)
cbar.set_label('Mean Activation-Loop Distance (A)')
ax.set_title('Activation-Loop Opening Distance Heatmap')
fig.tight_layout()
outfname=f'plots/egfr-activation-loop-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',dpi=600)
plt.close(fig)
print('analysis complete',flush=True)










