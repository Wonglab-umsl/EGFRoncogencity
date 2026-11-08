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

#These are the N-terminal and C-terminal portions of the dimer interface, 
#as determined from PDB code 3GOP by using all residues with at least one atom within 5 A of the other domain.
#dimer_int_n_term=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 58, 60, 64, 65, 67, 68, 71, 72, 84, 85, 86, 87, 88, 90])
#dimer_int_c_term=np.array([77, 116, 120, 124, 127, 153, 154, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 244, 245, 246, 248, 249, 252, 253, 255, 256, 257, 258, 261, 273, 276, 277, 278, 279, 280, 281, 284])

dimer_int_n_term="696 697 698 699 700 701 702 703 704 706 707 708 709 710 754 756 760 761 763 764 767 768 780 781 782 783 784 786"
dimer_int_c_term="773 812 816 820 823 849 850 926 927 928 929 930 931 932 933 934 935 937 940 941 942 944 945 948 949 951 952 953 954 957 969 972 973 974 975 976 977 980"


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


name_ids=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
average_sasa_dimer_int_n_term_df=pd.DataFrame(index=name_ids,columns=['Average SASA N-term'])
average_sasa_dimer_int_c_term_df=pd.DataFrame(index=name_ids,columns=['Average SASA C-term'])


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


#SASA is computed over the N-terminals and C-terminals
outfname=f'egfr-sasa-{mut_type}.txt'
output=open(outfname,'w')
output.write('Mutants\tSASA_N_term (A^2)\tSE_N_term\tSASA_C_term (A^2)\tSE_C_term\n')

for imut,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	#get the residues for this specific mutant
	dimer_int_n_term_atoms=top.select(f'protein and residue {dimer_int_n_term}')
	dimer_int_c_term_atoms=top.select(f'protein and residue {dimer_int_c_term}')
	reffname='data/{0}-solvated.pdb'.format(name)
	reftraj=md.load(reffname)

 	#Store data for each replica to calculate SE across replicas
	sasa_n_term_all_replicas=[]
	sasa_c_term_all_replicas=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#compute SASA using Shrake-Rupley algorithm
		#returns arrays of shape (n_frames, n_atoms) with SASA in A^2
		sasa=md.shrake_rupley(traj,n_sphere_points=96,mode='atom')

		#sum SASA over all atoms to get a total SASA per frame-- convert nm^2 to A^2
		sasa_dimer_int_n_term=np.sum(sasa[:,dimer_int_n_term_atoms],axis=1)*100
		sasa_dimer_int_c_term=np.sum(sasa[:,dimer_int_c_term_atoms],axis=1)*100

		#calculate mean SASA for this replica
		mean_sasa_n_term=np.mean(sasa_dimer_int_n_term)
		mean_sasa_c_term=np.mean(sasa_dimer_int_c_term)

		#pdb.set_trace()
		#calculation of SE across replicas
		sasa_n_term_all_replicas.append(mean_sasa_n_term)
		sasa_c_term_all_replicas.append(mean_sasa_c_term)

		#Store in DataFrame
		average_sasa_dimer_int_n_term_df.at[f'{name}-{id}','Average SASA N-term']=mean_sasa_n_term
		average_sasa_dimer_int_c_term_df.at[f'{name}-{id}','Average SASA C-term']=mean_sasa_c_term

		print(f'{name}-{id}: N-term SASA = {mean_sasa_n_term:.2f} A^2, C-term SASA = {mean_sasa_c_term:.2f} A^2',flush=True)

	#calculate the average and SE of the SASA across all replicas for each mutant
	sasa_n_term_all_replicas=np.array(sasa_n_term_all_replicas)
	sasa_c_term_all_replicas=np.array(sasa_c_term_all_replicas)

	overall_average_sasa_n_term=np.mean(sasa_n_term_all_replicas)
	overall_average_sasa_c_term=np.mean(sasa_c_term_all_replicas)

	se_sasa_n_term=np.std(sasa_n_term_all_replicas,ddof=1)/np.sqrt(nreplica)
	se_sasa_c_term=np.std(sasa_c_term_all_replicas,ddof=1)/np.sqrt(nreplica)

	output.write(f'{name}\t{overall_average_sasa_n_term:.2f}\t{se_sasa_n_term:.2f}\t{overall_average_sasa_c_term:.2f}\t{se_sasa_c_term:.2f}\n')
	print(f'{name}: N-term average SASA = {overall_average_sasa_n_term:.2f} +/- {se_sasa_n_term:.2f} A^2',flush=True)
	print(f'{name}: C-term average SASA = {overall_average_sasa_c_term:.2f} +/- {se_sasa_c_term:.2f} A^2',flush=True)

output.close()

#save the SASA data for each trajectory to csv file for use with random forests
average_sasa_dimer_int_n_term_df.to_csv('egfr-average-sasa-n-term.csv')
average_sasa_dimer_int_c_term_df.to_csv('egfr-average-sasa-c-term.csv')

print('SASA calculation complete',flush=True)

sys.exit()

#save the SASA data for each trajectory to csv file for use with random forests
average_sasa_dimer.to_csv('egfr-average-sasa.csv')

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


