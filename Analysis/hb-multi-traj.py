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
#import seaborn as sns
from operator import itemgetter
import pandas as pd
#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
scratch='/cluster/pixstor/wongch-lab/Victor/egfr//dcd-joined/'
#name=sys.argv[1]
nreplica=int(sys.argv[1])
stride=1
if (len(sys.argv)>2):
        stride=int(sys.argv[2])
#time between frames in ns
dt=0.001*stride 

distcutoff=2.5 #in A
anglecutoff=120 #in degrees

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

#pdb.set_trace()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
hbond_info={}
hbond_res_pairs=set()
#the dictionary is indexed by keys (name, donor_res, acc_res) and contain boolean array 
#in which frames have at least one hydrogen bond involving these residues in simulation of that system
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
		#use the md.baker_hubbard function to retrieve a list of hydrogen bonds that exist at least 1% of the time
		#then use top.atom() to convert the indices to atom objects and you can retrieve information (atom name, residue name, residue sequence number, etc.)
		#unfortunately it doesn't return a fraction for each hydrogen bond so if you want that information you will have to calcualte it separately
		#try using md.compute_distances and md.compute_angles and recheck the conditions for each hydrogen bond

		#If we are interested in only the side chain atoms for hydrogen bond analysis
		side_chain_atoms = traj.topology.select('sidechain')

		hbonds = md.baker_hubbard(traj, distance_cutoff=distcutoff/10, angle_cutoff=anglecutoff, freq=0.1, periodic=True)

		#When I need to filter the sidechain hydrogen bonds
		sidechain_hbonds = [(donor, hydrogen, acceptor) for (donor, hydrogen, acceptor) in hbonds if donor in side_chain_atoms and acceptor in side_chain_atoms]
		print('total {0:d} hydrogen bonds found'.format(len(sidechain_hbonds)),flush=True)
		#calculation of the fraction

		for ihbond, hbond in enumerate(sidechain_hbonds):
			donor=hbond[0]
			hydrogen=hbond[1]
			acceptor=hbond[2]
			#the residue objects are distinct for different trajectories even hough their string forms are not
			donor_res=(top.atom(donor).residue.code,top.atom(donor).residue.resSeq)
			acc_res=(top.atom(acceptor).residue.code,top.atom(acceptor).residue.resSeq)
			t=(name,id) + donor_res + acc_res #key into dictionary (name, donor code, donor res., acc. code, acc. res)
			hbond_res_pairs.add(donor_res + acc_res) #(donor code, donor res. number, acc. code, acc.res. number)
			#To compute distances between hydrogen and acceptor atoms and angles for the hydrogen bonds 
			distances = md.compute_distances(traj, [(hydrogen,acceptor)])*10 #nm to A
			angles = md.compute_angles(traj, [hbond])*180/np.pi #radians to degrees

			#Define hydrogen bond criteria:
			hbond_condition = np.all([(distances < distcutoff),(angles > anglecutoff)],axis=0)
			if (t in hbond_info):
				#logical "or" this with any hydrogen bonds involving the same residues
				hbond_info[t]=np.any([hbond_info[t],hbond_condition],axis=0)
			else:
				#put it in the dictonary
				hbond_info[t]=hbond_condition

		#we're done analyzing this trajectory, remove it from memory
		del traj
		print('{0:d} residue pairs found'.format(len(hbond_res_pairs)),flush=True)
		#pdb.set_trace()
		#pass

#sort the list of hydrogen bonds by donor res. number, then acc. res. number
hbond_res_pairs=sorted(hbond_res_pairs, key=itemgetter(1,3))

#Create a 2-D array of the occupancies of each hydrogen bond in each trajectory
occupancy=np.zeros((len(names)*nreplica,len(hbond_res_pairs)),dtype=np.float32)
for iname, name in enumerate(names):
	for id in range(1,nreplica+1):
		idx=iname*nreplica+id-1
		for ihbond, hbond in enumerate(hbond_res_pairs):
			#reconstruct key into dictionary
			t=(name,id)+hbond 
			#To calculate the fraction of frames where the hydrogen bond is present
			if (t in hbond_info):
				occupancy[idx,ihbond]=np.count_nonzero(hbond_info[t])/len(hbond_info[t])
			else:
				#this hydrogen bond is absent in this system, so the occupancy is zero by definition 
				#(this isn't strictly necessary, because we initialized to zeros to start with)
				occupancy[idx,ihbond]=0.0


hbond_labels=[]
for hbond in hbond_res_pairs:
        donor_code=hbond[0]
        donor_resnum=hbond[1] 
        acc_code=hbond[2]
        acc_resnum=hbond[3]
        #deal with fractional residue numbers for insertions
        hbond_label='{0}{1}-{2}{3}'.format(donor_code,donor_resnum,acc_code,acc_resnum) 
        hbond_labels.append(hbond_label)

#xticklabels=[labels[name] for name in names]

#finish and save the plot
#sns.set(font_scale=0.8)
#fig, ax = plt.subplots(figsize=(5, 10))
#cluster_map = sns.clustermap(occupancy, figsize=(20,5), cmap='Greys', row_cluster=True, col_cluster=True, xticklabels=hbond_labels, yticklabels=xticklabels,
#	dendrogram_ratio=(0.1,0.4))
#cluster_map.ax_heatmap.set_xlabel('residue pairs')
#cluster_map.ax_heatmap.set_ylabel('systems')
#cluster_map.ax_heatmap.set_aspect(0.5)
#outfname = 'plots/egfr-hbond-cluster-map.png'
#plt.savefig(outfname, format='png', bbox_inches='tight', dpi=600)
#plt.close()
#print(f'cluster-map saved as {outfname}', flush=True)



#to save as an excel or csv file
name_ids=[f'{name}-{id:d}' for name in names for id in range(1,nreplica+1)]
occupancy_df=pd.DataFrame(data=occupancy,index=name_ids,columns=hbond_labels)
excelfname='egfr-hbond-info.xlsx'
occupancy_df.to_excel(excelfname)
csvfname='egfr-hbond-info.csv'
occupancy_df.to_csv(csvfname)
