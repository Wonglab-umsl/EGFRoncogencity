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
import seaborn as sns
import pandas as pd

#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
scratch='/cluster/pixstor/wongch-lab/Victor/egfr//dcd-joined/'
#name=sys.argv[1]
nreplica=int(sys.argv[1])
stride=1
if (len(sys.argv)>1):
        stride=int(sys.argv[2])
#time between frames in ns
dt=0.001*stride 
distcutoff=4 #in A
fraccutoff=0.25 #this is the fraction of frames that must have a contact for it to be counted
#we will add the names of the mutants later

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




fig=plt.figure()
ax=fig.add_subplot(1,1,1)
contact_info={}
overall_res_pairs=set()
for name in names:
	#load everything
	psffname='psf/{0}.psf'.format(name)
	top=md.load_psf(psffname)
	#reffname='data/{0}-solvated.pdb'.format(name)
	#reftraj=md.load(reffname)
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'.format(scratch,name)
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)
		print('finished loading trajectory',flush=True)
		distances, res_pairs = md.compute_contacts(traj, contacts='all', scheme='closest-heavy')
		distances *= 10 #nm to A
		#pdb.set_trace()
		cont_frac=np.count_nonzero(distances<=distcutoff,axis=0)/traj.n_frames #fraction of frames within cutoff
		for i, (ires1, ires2) in enumerate(res_pairs):
			#exclude contacts that are too close in sequence (<=5 residues) and in which either residue is in the deletion area
			if ((ires2-ires1)>5):
				# and not ((ires1>=747 and ires1<=759) or (ires2>=747 and ires2<=759))):
				#identify each residue by a string 
				res1=top.residue(ires1).resSeq
				res2=top.residue(ires2).resSeq
				#exclude pairs in which either is in the deletion area
				if (not ((int(res1)>=747 and int(res1)<=759) or (int(res2)>=747 and int(res2)<=759))):
					res1=str(res1)
					res2=str(res2)
					overall_res_pairs.add((res1,res2))
					t=(name,id,res1,res2)
					contact_info[t]=cont_frac[i]

#overall_res_pairs is a set of tuples that include every residue pair considered in all the systems
#contact_info gives the contact fraction for each such pair and for each system
#create a 2D array with systems on one axis and residue pairs on the other
overall_res_pairs=sorted(list(overall_res_pairs))
cont_frac_2d = np.zeros((len(names),nreplica,len(overall_res_pairs)),dtype=np.float32)
for iname , name in enumerate(names):
	for id in range(1,nreplica+1):
		for ipair, pair in enumerate(overall_res_pairs):
			#Victor fills in the remainder of this loop, then re-enables and sets up the rest of the script to produce the contact map
			t = (name,id) + pair
			if t in contact_info:
			    cont_frac_2d[iname, id-1, ipair] = contact_info[t]

name_ids=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
#save the contact information before filtration
xticklabels=["{0}-{1}".format(pair[0], pair[1]) for pair in overall_res_pairs]
yticklabels=name_ids #[ for name in names]

#Create a pandas DataFrame for easy saving
df = pd.DataFrame(cont_frac_2d.reshape(len(names)*nreplica,len(overall_res_pairs)), index=yticklabels, columns=xticklabels)

#Save as CSV
df.to_csv("egfr-full-contact-matrix.csv")

#pdb.set_trace()
#remove contacts that are highly present in all the systems f
#find the minimum contact fraction for each contact over all the systems
min_cont_frac_2d=np.min(cont_frac_2d,axis=(0,1))
max_cont_frac_2d=np.max(cont_frac_2d,axis=(0,1))
#keep contacts which are greater than fraccutoff in at least one system and less than 1-fraccutoff (from 0.1 to 0.9)
contacts_keep=np.where(np.all([max_cont_frac_2d>=fraccutoff,min_cont_frac_2d<=1-fraccutoff],axis=0))[0]
#retain only the corresponding columns from contacts_keep
cont_frac_2d=cont_frac_2d[:,:,contacts_keep]
#retain the corresponding pairs in overall_res_pairs
overall_res_pairs=[overall_res_pairs[i] for i in contacts_keep]

print('{0:d} residue pairs detected'.format(len(overall_res_pairs)),flush=True)
#then use ax.imshow() to plot this as a contact-cluster-map

#come up with something for the x tick labels
xticklabels=["{0}-{1}".format(pair[0], pair[1]) for pair in overall_res_pairs]
yticklabels=name_ids #[ for name in names]

#Create a pandas DataFrame for easy saving
df = pd.DataFrame(cont_frac_2d.reshape(len(names)*nreplica,len(overall_res_pairs)), index=yticklabels, columns=xticklabels)

#Save as CSV
df.to_csv("egfr-reduced-contact-matrix.csv")

