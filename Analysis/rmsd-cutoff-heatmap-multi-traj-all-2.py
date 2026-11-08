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
rmsd_cutoff=float(sys.argv[3])
stride=1
if (len(sys.argv)>4):
        stride=int(sys.argv[4])
#time between frames in ns
dt=0.001*stride 
#scratch='/storage/hpc/group/wong/scratch/Victor/egfr/dcd-nowater'
scratch='/cluster/pixstor/wongch-lab/Victor/egfr//dcd-joined/'
#we will add the names of the mutants later


#RMSD cutoff
#rmsd_cutoff=2.0 #in Angstroms


#input=open(f'list-all','r')
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

rmsd_within_cutoff_data_list=[]
#Add list to store mean rmsd and fraction of time below cutoff for each trajectory
rmsd_mean_list=[]
rmsd_fraction_below_cutoff=[]

rmsd_mean_within_cutoff_list=[]

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
	fraction_below_cutoff_this_mutant=[]
	mean_rmsd_this_mutant=[]
	mean_rmsd_within_cutoff_this_mutant=[]
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#RMSD align trajectory against reference
		backbone_atoms=top.select('(name N or name CA or name C) and (residue 705 to 974)')
		traj.superpose(reftraj,atom_indices=backbone_atoms)
	
		#compute RMSDs -- convert nm to A
		rmsd=md.rmsd(traj,reftraj,atom_indices=backbone_atoms)*10
                #rmsd=md.rmsd(traj,reftraj,atom_indices=backbone_atoms)*10
		#rmsd=md.rmsd(traj,traj[0],atom_indices=backbone_atoms)*10
		#rmsd_data[imut,:]=rmsd
		rmsd_data_list.append(rmsd)

		#Calculate and store mean RMSD
		mean_rmsd=np.mean(rmsd)
		rmsd_mean_list.append(mean_rmsd)
		mean_rmsd_this_mutant.append(mean_rmsd)

		#Calculate fraction of frames below cutoff
		n_frames_below_cutoff=np.count_nonzero(rmsd<rmsd_cutoff)
		fraction_below_cutoff=n_frames_below_cutoff/len(rmsd)
		rmsd_fraction_below_cutoff.append(fraction_below_cutoff)
		fraction_below_cutoff_this_mutant.append(fraction_below_cutoff)
		print(f'{name}-{id}: {fraction_below_cutoff*100:.1f}% of frames below {rmsd_cutoff:.1f} A cutoff',flush=True)

		#Calculate mean RMSD within cutoff
		rmsd_within_cutoff=rmsd[rmsd<rmsd_cutoff]
		if len(rmsd_within_cutoff)>0:
			mean_rmsd_within_cutoff=np.mean(rmsd_within_cutoff)
			std_rmsd_within_cutoff=np.std(rmsd_within_cutoff)
		else:
			mean_rmsd_within_cutoff=np.nan
			std_rmsd_within_cutoff=np.nan
		rmsd_mean_within_cutoff_list.append(mean_rmsd_within_cutoff)
		mean_rmsd_within_cutoff_this_mutant.append(mean_rmsd_within_cutoff)
		#Store RMSD within cutoff for heatmap
		rmsd_within_cutoff_data_list.append(rmsd_within_cutoff)
		print(f'{name}-{id}: Mean RMSD within cutoff: {mean_rmsd_within_cutoff:.2f} +/- {std_rmsd_within_cutoff:.2f} A',flush=True)


	#Calculate overall mean and SE of the mean RMSD for each mutant
	mean_rmsd_this_mutant=np.array(mean_rmsd_this_mutant)
	overall_mean_rmsd=np.mean(mean_rmsd_this_mutant)
	overall_se_rmsd=np.std(mean_rmsd_this_mutant,ddof=1)/np.sqrt(len(mean_rmsd_this_mutant))
	print(f'{name} overall: {overall_mean_rmsd:.2f} +/- {overall_se_rmsd:.2f} A',flush=True)


	#Calculate overall mean and SE of the fraction below cutoff for each
	fraction_below_cutoff_this_mutant=np.array(fraction_below_cutoff_this_mutant)
	overall_mean_fraction_below_cutoff=np.mean(fraction_below_cutoff_this_mutant)
	overall_se_fraction_below_cutoff=np.std(fraction_below_cutoff_this_mutant,ddof=1)/np.sqrt(len(fraction_below_cutoff_this_mutant))
	print(f'{name} overall: {overall_mean_fraction_below_cutoff*100:.2f}% +/- {overall_se_fraction_below_cutoff*100:.2f}% below {rmsd_cutoff:.1f} A cutoff',flush=True)

	#Calculate overall mean and SE of the mean RMSD within cutoff for each mutant
	mean_rmsd_within_cutoff_this_mutant=np.array(mean_rmsd_within_cutoff_this_mutant)
	overall_mean_rmsd_within_cutoff=np.nanmean(mean_rmsd_within_cutoff_this_mutant)
	overall_se_rmsd_within_cutoff=np.nanstd(mean_rmsd_within_cutoff_this_mutant,ddof=1)/np.sqrt(np.sum(~np.isnan(mean_rmsd_within_cutoff_this_mutant)))
	print(f'{name} overall within cutoff: {overall_mean_rmsd_within_cutoff:.2f} +/- {overall_se_rmsd_within_cutoff:.2f} A',flush=True)


# Find the maximum number of frames across all trajectories
max_frames = max([len(rmsd) for rmsd in rmsd_data_list])

# Create a 2D array with NaN padding for RMSD data
rmsd_data=np.zeros((len(rmsd_data_list), max_frames), dtype=np.float32) 
#here imut actually indexes over mutation and replica
for imut, rmsd in enumerate(rmsd_data_list):
	rmsd_data[imut, :len(rmsd)] = rmsd

#Find the maximum number of frames within cutoff and create 2D array with NaN padding
max_frames_within_cutoff = max([len(rmsd) for rmsd in rmsd_within_cutoff_data_list])
rmsd_within_cutoff_data=np.full((len(rmsd_within_cutoff_data_list), max_frames_within_cutoff), np.nan, dtype=np.float32)
for imut, rmsd_within_cutoff in enumerate(rmsd_within_cutoff_data_list):
	rmsd_within_cutoff_data[imut, :len(rmsd_within_cutoff)] = rmsd_within_cutoff



print('RMSD range {0:.3f} - {1:.3f}'.format(np.min(rmsd_data),np.max(rmsd_data)),flush=True)
#the vmin and vmax give the colorbar range
_=ax.matshow(rmsd_within_cutoff_data,cmap='Greys',vmin=0,vmax=rmsd_cutoff)
cb=fig.colorbar(_,shrink=0.8,aspect=10)
cb.set_label(label='C$\\alpha$ RMSD ($\\AA$)',fontsize=11)
#finish and save the plot
ax.set_xlim(left=0,right=1000)
ax.set_xlabel('time (ns)')
ax.set_aspect(10)
if (mut_type=='pathogenic'):
	ax.set_aspect(5)
#this should ensure that the labels are in the same order as the names and in the correct positions
yticklabels=[]
for name in names:
	for id in range(1,nreplica+1):
		yticklabels.append(f'{labels[name]} rep. {id:d}')
ax.set_ylabel('backbone RMSD ($\\AA$)')
ax.set_yticks(np.arange(len(rmsd_within_cutoff_data_list)),yticklabels)
if (mut_type=='benign'):
	ax.tick_params(axis='y',labelsize=11)
elif (mut_type=='pathogenic'):
	ax.tick_params(axis='y',labelsize=5)
ax.set_yticklabels(yticklabels)
outfname=f'plots/egfr-rmsd-within-cutoff-heatmap-{mut_type}.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)


#Write fraction below cutoff to output file with mean and SE for each mutant
outfname_cutoff=f'egfr-rmsd-fraction-below-{rmsd_cutoff:.1f}A-reference-{mut_type}.txt'
output_cutoff=open(outfname_cutoff,'w')
output_cutoff.write(f'RMSD cutoff: {rmsd_cutoff:.1f} A\n')
output_cutoff.write(f'RMSD measured relative to solvated reference structure (experimentally relevant)\n')
output_cutoff.write('Simulation\tFraction below cutoff\tPercentage below cutoff\tMean RMSD (A)\n')
for (name, id), fraction in zip([(n, i) for n in names for i in range(1, nreplica+1)], rmsd_fraction_below_cutoff):
	output_cutoff.write(f'{name}-{id}\t{fraction:.4f}\t{fraction*100:.2f}%\n')

#statistics for each mutant
output_cutoff.write('\n--- Summary Statistics by Mutant ---\n')
output_cutoff.write('Mutant\tMean Fraction Below Cutoff\tSE Fraction Below Cutoff\tMean % Below Cutoff\tSE % Below Cutoff\tMean RMSD (A)\tSE Mean RMSD (A)\n')
idx=0
for imut,name in enumerate(names):
	fractions_this_mutant=np.array(rmsd_fraction_below_cutoff[idx:idx+nreplica])
	mean_fraction=np.mean(fractions_this_mutant)
	se_fraction=np.std(fractions_this_mutant,ddof=1)/np.sqrt(nreplica)
	#Calculate mean and SE of mean RMSD for each mutant
	mean_rmsd_values=np.array(rmsd_mean_list[idx:idx+nreplica])
	mean_of_mean_rmsd=np.mean(mean_rmsd_values)
	se_mean_rmsd=np.std(mean_rmsd_values,ddof=1)/np.sqrt(nreplica)
	#Calculate mean and SE of mean RMSD within cutoff for each mutant
	mean_rmsd_within_cutoff_values=np.array(rmsd_mean_within_cutoff_list[idx:idx+nreplica])
	mean_of_mean_rmsd_within_cutoff=np.nanmean(mean_rmsd_within_cutoff_values)
	se_mean_rmsd_within_cutoff=np.nanstd(mean_rmsd_within_cutoff_values,ddof=1)/np.sqrt(np.sum(~np.isnan(mean_rmsd_within_cutoff_values)))
	output_cutoff.write(f'{name}\t{mean_fraction:.4f}\t{se_fraction:.4f}\t{mean_fraction*100:.2f}%\t{se_fraction*100:.2f}%\t{mean_of_mean_rmsd:.2f}\t{se_mean_rmsd:.2f}\t{mean_of_mean_rmsd_within_cutoff:.2f}\t{se_mean_rmsd_within_cutoff:.2f}\n')
	idx+=nreplica
output_cutoff.close()


#Create a CSV file
outfname_csv=f'egfr-rmsd-fraction-below-{rmsd_cutoff:.1f}A-reference-{mut_type}.csv'
csv_data=[]
for (name, id), fraction, mean_rmsd in zip([(n, i) for n in names for i in range(1, nreplica+1)], rmsd_fraction_below_cutoff, rmsd_mean_list):
	csv_data.append({'': name,'mean RMSD': fraction})
	#csv_data.append({'Mutant': name, 'Replica': id, 'Fraction_Below_Cutoff': fraction,'Percentage_Below_Cutoff': fraction*100, 'Mean_RMSD_A': mean_rmsd})
csv_df=pd.DataFrame(csv_data)
csv_df.to_csv(outfname_csv,index=False)
print(f'CSV file saved: {outfname_csv}',flush=True)


#Create a separate CSV file with only mutants and mean RMSD within cutoff
name_ids_mutants=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
mean_rmsd_within_cutoff_df=pd.DataFrame(index=name_ids_mutants,columns=['mean RMSD'])
for i, name_id in enumerate(name_ids_mutants):
	mean_rmsd_within_cutoff_df.at[name_id,'mean RMSD']=rmsd_mean_within_cutoff_list[i]

outfname_csv_mean=f'egfr-mean-rmsd-within-cutoff-{mut_type}.csv'
mean_rmsd_within_cutoff_df.to_csv(outfname_csv_mean)
print(f'CSV file with mean RMSD within cutoff saved: {outfname_csv_mean}',flush=True)
