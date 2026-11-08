#!/usr/bin/env python

import sys
import math
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
import pickle
import pdb

#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined/'
nreplica=int(sys.argv[1])
stride=int(sys.argv[2])
cutoff=float(sys.argv[3])/10
ncomponents=int(sys.argv[4])

dt=0.001*stride #in ns

def read_name_list(fname):
	input=open(fname,'r')
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
	return labels

#The initial PCA analysis should be done using all the mutants
labels=read_name_list('list1')
names=labels.keys()


trajlist=[]
frame_start={}
frame_end={}
counter=0
for name in labels.keys():
	#load everything
	psffname='psf/{0}.psf'.format(name)
	top=md.load_psf(psffname)
	reffname='data/egfr-wt-active-charmm-solvated.pdb'.format(name)
	reftraj=md.load(reffname)

	#RMSD align trajectory against reference
	ca_atoms=top.select('name CA and (residue 705 to 974) and not (residue 746 to 759)')
	#remove any inserted residues (non-integer residue numbers) -- see psf.py hack in mdtraj
	ca_atoms2=[]
	for atom_idx in ca_atoms:
		res=top.atom(atom_idx).residue.resSeq
		if (res==int(res)):
			ca_atoms2.append(atom_idx)
	ca_atoms=ca_atoms2
	#We append all the individual replicates for each system to get one single trajectory for that system.
	trajlist_this_system=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)
		#print(ca_atoms)
		traj.superpose(reftraj,atom_indices=ca_atoms)
		#strip out only the alpha carbons common to all three trajectories
		traj=traj.atom_slice(atom_indices=ca_atoms)
		print(traj.n_atoms)
		trajlist_this_system.append(traj)

		traj=md.join(trajlist_this_system)
		trajlist.append(traj)
		print(f'total frames for {name} id {id}: {traj.n_frames}',flush=True)
		#keep track of which frames came from which system
		frame_start[(name,id)]=counter
		counter+=traj.n_frames
		frame_end[(name,id)]=counter-1


#join the trajectories for each system together. 
#mdtraj checks to see if the residues are the same for all of them (they are not because of the mutations)
#it is ok because we are only working with alpha carbons.  disable the check
alltraj=md.join(trajlist,check_topology=False)
del trajlist
print("total frames: ",alltraj.n_frames,flush=True)
print(frame_start,flush=True)
print(frame_end,flush=True)

#perform the PCA in terms of interatomic distances between alpha carbons.
all_atom_pairs = list(combinations(range(alltraj.n_atoms), 2))
#filter out only those pairs where the residues are not neighboring
new_pairs=[]
for iatom, jatom in all_atom_pairs:
        ires=traj.topology.atom(iatom).residue.resSeq
        jres=traj.topology.atom(jatom).residue.resSeq
        if ((jres-ires)>1):
                new_pairs.append((iatom,jatom))

pairwise_distances = md.geometry.compute_distances(alltraj, new_pairs) #of shape(n_frames, n_pairs)
min_pairwise_distances = np.min(pairwise_distances, axis=0) #this is the minimum distance encountered over the whole trajectory

#the np.where function gets a list of indices where the entry satisfies a condition
#here it is the indices of atom pairs where the distance is ever less than the cutoff.
close_pair_indices=list(np.where(min_pairwise_distances<cutoff)[0])

#convert from indices to actual pairs
atom_pairs=[new_pairs[ipair] for ipair in close_pair_indices]
n_pairs=len(atom_pairs)
print("total number of atom pairs: ",n_pairs,flush=True)

pairwise_distances = pairwise_distances[:,close_pair_indices]
print(pairwise_distances.shape)

#perform principal component analysis
pca=PCA(n_components=ncomponents)
pca_coords = pca.fit_transform(pairwise_distances)
	
print(pca_coords.shape) 
print(pca.explained_variance_ratio_)


#create and plot a heatmap of the components
comps=pca.components_[0:ncomponents,:]
print(comps)
max_ev_comp=np.max(np.abs(comps))
print("maximum eigenvector component: ",max_ev_comp,flush=True)
nres=len(ca_atoms)
comp_by_res=np.zeros((ncomponents,nres,nres),dtype=np.float32)
resnums=[]
for res in range(0,nres):
	idx=ca_atoms[res]
	resnums.append(top.atom(idx).residue.resSeq)

for ipair, pair in enumerate(atom_pairs):
	iatom=pair[0]
	jatom=pair[1]
	comp_by_res[:,iatom,jatom]=comps[:,ipair]

print(np.sum(comp_by_res**2,axis=(1,2)))
cmap=matplotlib.colormaps.get_cmap('RdBu') #('gist_gray_r')
xticks=range(0,nres,50)
xticklabels=resnums[::50]
yticks=range(0,nres,20)
yticklabels=resnums[::20]
for icomp in range(0,ncomponents):
	print(f'plotting eigenvector for component {icomp+1:d}',flush=True)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.set_aspect('equal')
	_=ax.imshow(comp_by_res[icomp,:,:],cmap=cmap,vmin=-0.1,vmax=0.1)
	ax.set_xlabel('residue number')
	ax.set_xticks(xticks)
	ax.set_xticklabels(xticklabels)
	ax.set_ylabel('residue number')
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticklabels)
	ax.axline((0,0),(nres,nres),color='black')
	fig.colorbar(_)
	outfname=f'plots/egfr-pca-distances-reduced-all-data-comp-{icomp+1:d}.png'
	fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
	plt.close(fig)

	#make a file containing a list of the atom pairs and eigenvector components, sorted by square of the eigenvector component 
	#it will be more convenient to work with comps[:,ipair]
	outfname=f'egfr-pca-distances-reduced-all-data-comp-{icomp+1:d}'
	output=open(outfname,'w')
	order=np.flip(np.argsort(comps[icomp,:]**2)) #list of pair indices corresponding to eignvector components in order 
	for ipair in order:
		iatom=atom_pairs[ipair][0]
		ires=alltraj.topology.atom(iatom).residue.resSeq
		jatom=atom_pairs[ipair][1]
		jres=alltraj.topology.atom(jatom).residue.resSeq
		output.write('{0:d} {1:d} {2:.6f} {3:.6f}\n'.format(ires,jres,comps[icomp,ipair],comps[icomp,ipair]**2))
	output.close()

#print(resnums)

#save frames to a binary file
output=open('egfr-pca-distances-reduced-all-data2.p','wb')
pickle.dump(frame_start,output)
pickle.dump(frame_end,output)
pickle.dump(pca_coords,output)
output.close()


#print(resnums)


