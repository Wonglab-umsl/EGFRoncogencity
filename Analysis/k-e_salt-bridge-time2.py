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
from collections import defaultdict
import pdb
scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined'
dt=0.001 #in ns



mut_type=sys.argv[1]
nreplica=int(sys.argv[2])
res_start=int(sys.argv[3])
res_end=int(sys.argv[4])
nres=res_end-res_start+1
stride=1
if (len(sys.argv)>5):
        stride=int(sys.argv[5])


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

occupancy_dict=defaultdict(list)

fig=plt.figure(constrained_layout=True,figsize=(12,len(names)*nreplica*1.5))
#spec=gridspec.GridSpec(nrows=len(names),ncols=nres,figure=fig)
ticks=np.linspace(start=0,stop=10,num=6)
blank_labels=['']*len(ticks)
for iname,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)


	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#Calculate K-E salt-bridge distance using res_start and res_end
		#res_start is lysine residue, res_end is glutamic acid residue
		k_selection=f'residue {res_start} and name NZ'
		e_selection=f'residue {res_end} and (name OE1 or name OE2)'
		k_atoms=top.select(k_selection)
		e_atoms=top.select(e_selection)

		if len(k_atoms)==0 or len(e_atoms)==0:
			print(f'Warning: missing atoms for {name}-{id}')
			continue

		#Build all NZ-OE distance pairs
		pairs=[[k_atoms[0],e] for e in e_atoms]
		#Calculate distance between K and E
		distances=md.compute_distances(traj,pairs)
		#Take minimum NZ-OE distance for each frame
		ke_distance=np.min(distances,axis=1)*10 #convert nm to Angstroms
		print(f'{name}-{id}: K-E distance shape: {ke_distance.shape}',flush=True)

		#construct subplot index
		subplot_index=1+(iname*nreplica+id-1)
		ax=fig.add_subplot(len(names)*nreplica,1,subplot_index)

		#color by fraction of the trajectory
		#time=np.linspace(start=0,stop=1,num=traj.n_frames)
		frames=np.arange(traj.n_frames)
		time_ns=frames*dt*stride

		#scatter=ax.scatter(time_ns,ke_distance,marker='.',s=5,c=time,cmap='gray')

		ax.plot(time_ns,ke_distance,linewidth=0.5,color='black',alpha=0.9)
		ax.axhline(y=4.0,color='black',linestyle='--',linewidth=1,alpha=0.7)
		ax.set_xlim(left=0,right=time_ns[-1])
		ax.set_ylim(bottom=2,top=6)
		ax.set_ylabel(f'{labels[name]} rep. {id:d}',fontsize=8)
		ax.tick_params(labelsize=8)
		ax.grid(alpha=0.3)
		#ax.axhline(y=4.0,color='black',linestyle='--',linewidth=1)

		if (iname==len(names)-1):
			ax.set_xlabel('time (ns)',fontsize=10)
		if (iname==0):
			ax.set_title(f'K-E Salt-Bridge Distance (res {res_start}-{res_end}) Time Course',fontdict={'fontsize':12})

		#Calculate salt bridge occupancy
		occupancy=np.mean(ke_distance < 4.0)
		occupancy_dict[name].append(occupancy)
		print(f'{name}-{id}: occupancy = {occupancy:.3f}',flush=True)




#Summary statistics (MEAN ± SEM)
print("\n=== Salt-bridge occupancy summary (MEAN ± SE) ===")

means = []
sems = []

for name in names:
	vals = np.array(occupancy_dict[name])
	mean = vals.mean()
	sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
	means.append(mean)
	sems.append(sem)
	print(f"{name}: mean={mean:.3f}, SEM={sem:.3f}, n={len(vals)}")


#Bar plot with SEM
fig2, ax2 = plt.subplots(figsize=(6, 4))
xpos = np.arange(len(names))
ax2.bar(xpos, means, yerr=sems, capsize=5, alpha=0.6)
ax2.set_xticks(xpos)
ax2.set_xticklabels(names, rotation=45, ha='right')
ax2.set_ylabel("K745–E762 occupancy (<4 Å)")
ax2.set_ylim(0, 1)
fig2.tight_layout()
fig2.savefig(f'plots/egfr-ke-occupancy-{mut_type}-{res_start}-{res_end}.png', dpi=600)

#finish and save the plot
fig.subplots_adjust(wspace=0,hspace=0.3)
outfname=f'plots/egfr-ke-saltbridge-distance-{mut_type}-{res_start:d}-{res_end:d}-bytime.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)

plt.close(fig)
plt.close(fig2)

