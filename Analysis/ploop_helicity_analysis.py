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
import csv
import pdb
from collections import defaultdict
#from scipy.spatial.distance import squareform
#from datetime import datetime as dtt
import pickle
#scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined'

def count_consecutive(ss_sequence, ss_type):
	count=0
	for i in range(0,len(ss_sequence)):
		if ((ss_sequence[i]==ss_type) and ((i==0) or (ss_sequence[i-1]!=ss_type))):
			count+=1
	return count

#pdb.set_trace()
print(count_consecutive('HHHHCCCCHHHH','H')) #HHCCCCHH
print(count_consecutive('CHHHHCCC','H')) #CHHCCC


#sys.exit(0)
dt=0.001 #in ns
mut_type=sys.argv[1]
nreplica=int(sys.argv[2])
stride=1
if (len(sys.argv)>3):
        stride=int(sys.argv[3])

# Load secondary structure data from pickle
input_file = open('/cluster/pixstor/wongch-lab/Victor/egfr/egfr-sec-struc-info.p', 'rb')
ss = pickle.load(input_file)
input_file.close()

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


scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined' 
AL_START = 855
AL_END = 871
PL_START = 718
PL_END = 726


# Use {'H', 'G', 'I'} if ss contains original 8-state DSSP assignments.
HELIX_CODES = {'H'}

residue_to_index = {}


for name in names:
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)

	# Get CA atoms and their residue numbers
	ca_atoms = top.select('name CA')
	resnums = []
	for idx in ca_atoms:
		resnums.append(top.residue(top.atom(idx).residue.index).resSeq)

	# Map AL residue numbers to indices
	residue_to_index[name] = {}
	for array_idx, resnum in enumerate(resnums):
		residue_to_index[name][resnum] = array_idx


	print(f"{name}: PL residues {PL_START}-{PL_END} map to array indices {[residue_to_index[name].get(r, 'N/A') for r in range(PL_START, PL_END+1)]}")
	break  # PSF topology should be same for all replicates, use first name



#Analyze turn presence in the activation loop
def detect_helical_turn(ss_sequence, min_helix_residues=4, max_helix_residues=4):
	"""
	Detect presence of 1-2 helical residues in a secondary structure sequence.
	A 'turn' is defined as 1-2 consecutive helical residues ('H')
	flanked by non-helical residues (coil or sheet).

	Returns: True if turn detected, False otherwise
	"""
	if len(ss_sequence) < 1:
		return False

	# Count consecutive helices
	helix_count = 0
	for i, ss in enumerate(ss_sequence):
		if ss == 'H':
			helix_count += 1
		else:
			if min_helix_residues <= helix_count <= max_helix_residues:
				#Check that helix is flanked by non-helix (not at boundaries)
				#if i - helix_count > 0 and i < len(ss_sequence):
				return True
			helix_count=0

	#check final helix block
	if min_helix_residues <= helix_count <= max_helix_residues:
		#if len(ss_sequence) - helix_count > 0:
		return True

	return False

def calculate_helix_content(ss_sequence):
	"""Calculate fraction of helical residues"""
	if len(ss_sequence) == 0:
		return np.nan
	return np.sum(np.array(ss_sequence) == 'H') / len(ss_sequence)

#Extract AL residue indices for each system
# EDIT: Return both topology indices and their actual PDB residue numbers.
def get_region_indices(psf_top, region_start, region_end, region_name):
	"""Return SS-array indices and PDB residue numbers for one protein region."""
	ca_atoms = psf_top.select('name CA')

	ca_resnums = np.array([
		psf_top.residue(psf_top.atom(atom_idx).residue.index).resSeq
		for atom_idx in ca_atoms])
	region_mask = (ca_resnums >= region_start) & (ca_resnums <= region_end)

	region_indices = np.where(region_mask)[0].tolist()
	region_resnums = ca_resnums[region_mask].tolist()

	if len(region_indices) == 0:
		raise ValueError(
			f'No CA residues found between {region_start} and {region_end}')
	return region_indices, region_resnums






#def get_al_indices(name, psf_top, AL_START, AL_END):
#	"""Get array indices for AL residues"""
#	ca_atoms = psf_top.select('name CA')
#	resnums = []
#	for idx in ca_atoms:
#		resnums.append(psf_top.residue(psf_top.atom(idx).residue.index).resSeq)

#	al_indices = []
#	for resnum in range(AL_START, AL_END + 1):
#		if resnum in resnums:
#			al_indices.append(resnums.index(resnum))

#	return al_indices

summary = []
turn_presence_data = []
turn_presence_labels = []
per_residue_summary = []
pl_per_residue_summary = []

for iname, name in enumerate(names):
	print(f"\nProcessing system: {name}")

	#Load PSF for this system
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	al_indices, al_resnums = get_region_indices(top, AL_START, AL_END, 'A-loop')
	pl_indices, pl_resnums = get_region_indices(top, PL_START, PL_END, 'P-loop')

#	al_indices = get_al_indices(name, top, AL_START, AL_END)

	print(f"A-loop residues: {al_resnums};" f"SS indices: {al_indices}")
	print(f"P-loop residues: {pl_resnums};" f"SS indices: {pl_indices}")

	for id in range(1, nreplica + 1):
		print(f"  Analyzing replica {id}...")
		# Get secondary structure for this trajectory
		if (name, id) not in ss:
			print(f" Warning: No SS data for {name}-{id}")
			continue
		ss_data = ss[(name, id)][::stride]  # Apply stride if needed
		nframes = ss_data.shape[0]

		print(f" Total frames: {nframes}")

		# Extract AL secondary structure
		al_ss = ss_data[:, al_indices]  # (n_frames, n_al_residues)
		pl_ss = ss_data[:, pl_indices]  # (n_frames, n_pl_residues
		#pdb.set_trace()

		# Calculate metrics for each frame
		nturn_helix = []
		nturn_sheet = []
		nhelix = []
		nsheet = []
		nshort_helix = []


		for frame_idx in range(nframes):

			ss_frame = al_ss[frame_idx, :]
			nturn_helix.append(count_consecutive(ss_frame,'H'))
			nturn_sheet.append(count_consecutive(ss_frame,'E'))

			# Calculate fractions
			nhelix.append(np.sum(ss_frame == 'H') / len(ss_frame))
			nsheet.append(np.sum(ss_frame == 'E') / len(ss_frame))

			#detect short helix turn
			has_short_helix = detect_helical_turn(ss_frame, min_helix_residues=3, max_helix_residues=7)
			nshort_helix.append(1 if has_short_helix else 0)

		nturn_helix = np.array(nturn_helix)
		nturn_sheet = np.array(nturn_sheet)
		nhelix = np.array(nhelix)
		nsheet = np.array(nsheet)
		nshort_helix = np.array(nshort_helix)

		# np.isin permits either simplified {'H'} or full DSSP {'H', 'G', 'I'}.
		per_residue_helix_occ = np.mean(np.isin(al_ss, list(HELIX_CODES)),axis=0)
		pl_per_residue_helix_occ = np.mean(np.isin(pl_ss, list(HELIX_CODES)),axis=0)
		#per_residue_helix_occ = np.mean(al_ss == 'H', axis=0)
		per_residue_sheet_occ = np.mean(al_ss == 'E', axis=0)

		for residue, helix_occ, sheet_occ in zip(al_resnums,per_residue_helix_occ,per_residue_sheet_occ):
			per_residue_summary.append({'system': name,'label': labels[name],'replica': id,'residue': residue,'helix_occ': helix_occ,'sheet_occ': sheet_occ})
		for residue, helix_occ, sheet_occ in zip(pl_resnums,pl_per_residue_helix_occ,per_residue_sheet_occ):
			pl_per_residue_summary.append({'system': name,'label': labels[name],'replica': id,'residue': residue,'helix_occ': helix_occ,'sheet_occ': sheet_occ})



#		for i, occ in enumerate(per_residue_helix_occ):
#			resnum = AL_START + i

#			per_residue_summary.append({'system': name,'replica': id,'residue': resnum,'helix_occ': occ,'sheet_occ': per_residue_sheet_occ[i]})

		# Calculate summary statistics
		helix_turn_occupancy = np.mean(nturn_helix)
		sheet_turn_occupancy = np.mean(nturn_sheet)
		helix_fraction_occupancy = np.mean(nhelix)
		sheet_fraction_occupancy = np.mean(nsheet)
		helix_turn_occupancy_se = np.std(nturn_helix) / np.sqrt(len(nturn_helix))
		short_helix_occupancy = np.mean(nshort_helix)
		short_helix_occupancy_se = np.std(nshort_helix) / np.sqrt(len(nshort_helix))



		summary.append({
			'system': name,
			'replica': id,
			'helix_turn_count': helix_turn_occupancy,
			'helix_turn_count_se': helix_turn_occupancy_se,
			'sheet_turn_count': sheet_turn_occupancy,
			'helix_fraction': helix_fraction_occupancy,
			'short_helix_occupancy': short_helix_occupancy,
			'short_helix_occupancy_se': short_helix_occupancy_se,
			'sheet_fraction': sheet_fraction_occupancy,
			'n_frames': nframes
		})

		print(f"{name}-{id}: Mean helix blocks = {helix_turn_occupancy:.3f}, "f"3–7-residue helix occupancy = {short_helix_occupancy:.3f},"f"Helix fraction = {helix_fraction_occupancy:.3f}")

		turn_presence_data.append(nturn_helix)
		turn_presence_labels.append(f'{labels[name]} rep. {id:d}')
		turn_presence_data.append(nshort_helix)
		turn_presence_labels.append(f'{labels[name]} rep. {id:d} (3-7 helix present)')

#Main analysis plot — A-loop helical occupancy by residue
helix_by_system_replica = defaultdict(lambda: defaultdict(dict))
for row in per_residue_summary:
        helix_by_system_replica[row['system']][row['replica']][row['residue']] = row['helix_occ']

global_upper = 0.0

fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(helix_by_system_replica)))
for color, system in zip(colors, sorted(helix_by_system_replica)):
        replica_data = helix_by_system_replica[system]
        # Use the common residue range across all replicas for this system
        common_residues = sorted(set.intersection(*[set(replica_data[replica].keys())for replica in replica_data]))

        # rows = replicas; columns = A-loop residues
        occupancy_matrix = np.array([[replica_data[replica][residue] for residue in common_residues]for replica in sorted(replica_data)])
        mean_occupancy = np.mean(occupancy_matrix, axis=0)
        if occupancy_matrix.shape[0] > 1:
                se_occupancy = (np.std(occupancy_matrix, axis=0, ddof=1) /np.sqrt(occupancy_matrix.shape[0]))
        else:
                se_occupancy = np.zeros_like(mean_occupancy)
        global_upper = max(global_upper,np.nanmax(mean_occupancy + se_occupancy))
        # Show individual replicas faintly
        for replica_values in occupancy_matrix:
                ax.plot(common_residues,replica_values,color=color,alpha=0.20,linewidth=1.0)
        # Show system mean prominently
        ax.plot(common_residues,mean_occupancy,color=color,marker='o', markersize=4,linewidth=2.5,label=labels[system])

        # Shading represents SEM across independent replicas
        ax.fill_between(common_residues,np.clip(mean_occupancy - se_occupancy, 0, 1),np.clip(mean_occupancy + se_occupancy, 0, 1),color=color,alpha=0.18)


ax.set_xlabel('A-loop residue number')
ax.set_ylabel('Helical occupancy (%)')
ax.set_xlim(AL_START - 0.5, AL_END + 0.5)
ymax = min(1.0, max(0.01, np.ceil(global_upper * 100 / 1.0) / 100))
ax.set_ylim(0, ymax)
ax.set_yticks(np.linspace(0, ymax, 6))
ax.set_yticklabels([f'{100 * value:.0f}%' for value in np.linspace(0, ymax, 6)])

#ax.set_ylim(0, 0.03) #0-5% helical occupancy
#ax.set_yticks(np.arange(0, 0.031, 0.01))
#ax.set_yticklabels([f'{100*x:.0f}%' for x in np.arange(0, 0.031, 0.01)])


ax.set_xticks(np.arange(AL_START, AL_END + 1, 2))
ax.grid(alpha=0.25)
ax.legend(title='System', frameon=False, fontsize=9)
ax.set_title('Activation-Loop Helical Occupancy by Residue')

fig.tight_layout()
outfname = f'plots/egfr-al-per-residue-helix-{mut_type}.png'
fig.savefig(outfname, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved per-residue helix occupancy plot: {outfname}")


#system-by-residue mean helicity heatmp
systems = sorted(helix_by_system_replica)
all_residues = np.arange(AL_START, AL_END + 1)
system_residue_matrix = np.full((len(systems), len(all_residues)),np.nan)
for i, system in enumerate(systems):
        replica_data = helix_by_system_replica[system]
        for j, residue in enumerate(all_residues):
                values = [replica_data[replica][residue] for replica in replica_data if residue in replica_data[replica]]
                if len(values) > 0:
                        system_residue_matrix[i, j] = np.mean(values)

fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(systems))))
matrix_max = np.nanmax(system_residue_matrix)
heatmap_vmax = max(0.01, np.ceil(matrix_max * 100) / 100)
im = ax.imshow(system_residue_matrix,aspect='auto',interpolation='nearest',cmap='Greys',vmin=0,vmax=heatmap_vmax)
ax.set_xticks(np.arange(len(all_residues)))
ax.set_xticks(np.arange(len(all_residues)))
ax.set_yticks(np.arange(len(systems)))
ax.set_yticklabels([labels[system] for system in systems])

ax.set_xlabel('A-loop residue number')
ax.set_ylabel('System')
ax.set_title('Mean A-Loop Helical Occupancy')
cbar = fig.colorbar(im, ax=ax)
#cbar.set_ticklabels(['0%', '1%', '2%', '3%', '4%', '5%', '6%']) #'4%', '5%'
cbar.set_label('Mean helical occupancy')
#cbar.set_label('Fraction of frames assigned as helix')
fig.tight_layout()
outfname = f'plots/egfr-al-per-residue-helix-heatmap-{mut_type}.png'
fig.savefig(outfname, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved per-residue helix occupancy heatmap: {outfname}")




#A-loop sheet occupancy by residue
sheet_by_system_replica = defaultdict(lambda: defaultdict(dict))

for row in per_residue_summary:
        sheet_by_system_replica[row['system']][row['replica']][row['residue']] = row['sheet_occ']
fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(sheet_by_system_replica)))
for color, system in zip(colors, sorted(sheet_by_system_replica)):
        replica_data = sheet_by_system_replica[system]
        # Residues present in every replica of this system
        common_residues = sorted(set.intersection(*[set(replica_data[replica].keys())for replica in replica_data]))
        # One row per replica; one column per A-loop residue
        occupancy_matrix = np.array([[replica_data[replica][residue] for residue in common_residues]for replica in sorted(replica_data)])
        mean_occupancy = np.mean(occupancy_matrix, axis=0)

        if occupancy_matrix.shape[0] > 1:
                se_occupancy = (np.std(occupancy_matrix, axis=0, ddof=1) /np.sqrt(occupancy_matrix.shape[0]))
        else:
                se_occupancy = np.zeros_like(mean_occupancy)

        #Thin, translucent lines: individual replicas
        for replica_values in occupancy_matrix:ax.plot(common_residues,replica_values,color=color,alpha=0.20,linewidth=1.0)

        # Dark line: mean across replicas
        ax.plot(common_residues,mean_occupancy,color=color,marker='o',markersize=4,linewidth=2.5,label=labels[system])

        # Shading: SE across replicas
        ax.fill_between(common_residues,np.clip(mean_occupancy - se_occupancy, 0, 1),np.clip(mean_occupancy + se_occupancy, 0, 1),color=color,alpha=0.18)

ax.set_xlabel('A-loop residue number')
ax.set_ylabel('Sheet occupancy (%)')
ax.set_xlim(AL_START - 0.5, AL_END + 0.5)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(AL_START, AL_END + 1, 2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax.grid(alpha=0.25)

ax.legend(title='System',frameon=False,fontsize=6,bbox_to_anchor=(1.02, 1),loc='upper left')
ax.set_title('Activation-Loop Sheet Occupancy by Residue')
fig.tight_layout()
outfname = f'plots/egfr-al-per-residue-sheet-{mut_type}.png'

fig.savefig(outfname, dpi=600, bbox_inches='tight')
plt.close(fig)

print(f"Saved per-residue sheet occupancy plot: {outfname}")

#P-loop helical occupancy by residue
pl_helix_by_system_replica = defaultdict(lambda: defaultdict(dict))
for row in pl_per_residue_summary:
	pl_helix_by_system_replica[row['system']][row['replica']][row['residue']]=row['helix_occ']

fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(pl_helix_by_system_replica)))
global_upper = 0.0
for color, system in zip(colors,sorted(pl_helix_by_system_replica)):
	replica_data = pl_helix_by_system_replica[system]

	# Use residues represented in every replica for this system
	common_residues = sorted(set.intersection(*[set(replica_data[replica].keys())for replica in replica_data]))

	# Rows = replicas; columns = P-loop residues
	occupancy_matrix = np.array([[replica_data[replica][residue]for residue in common_residues]for replica in sorted(replica_data)])
	mean_occupancy = np.mean(occupancy_matrix, axis=0)

	if occupancy_matrix.shape[0] > 1:
		se_occupancy = (np.std(occupancy_matrix, axis=0, ddof=1) /np.sqrt(occupancy_matrix.shape[0]))
	else:
		se_occupancy = np.zeros_like(mean_occupancy)

	global_upper = max(global_upper,np.nanmax(mean_occupancy + se_occupancy))

	# Pale lines = individual replicas
	for replica_values in occupancy_matrix:
		ax.plot(common_residues,replica_values,color=color,alpha=0.20,linewidth=1.0)

	# Dark line = mean across replicas
	ax.plot(common_residues,mean_occupancy,color=color,marker='o',markersize=4,linewidth=2.5,label=labels[system])

	# Shaded band = SE across replicas
	ax.fill_between(common_residues,np.clip(mean_occupancy - se_occupancy, 0, 1),np.clip(mean_occupancy + se_occupancy, 0, 1),color=color,alpha=0.18)

# Set a scale appropriate to observed P-loop helicity
# The minimum 1% scale prevents an unusably narrow axis if helicity is tiny
ymax = min(1.0,max(0.01, np.ceil(global_upper * 100) / 100))
ax.set_xlabel('P-loop residue number')
ax.set_ylabel('Helical occupancy (%)')
ax.set_xlim(PL_START - 0.5, PL_END + 0.5)
ax.set_ylim(0, ymax)


yticks = np.linspace(0, ymax, 6)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{100 * value:.0f}%' for value in yticks])

ax.set_xticks(np.arange(PL_START, PL_END + 1, 2))
ax.grid(alpha=0.25)
ax.legend(title='System',frameon=False,fontsize=9,bbox_to_anchor=(1.02, 1),loc='upper left')
ax.set_title('P-Loop Helical Occupancy by Residue')
fig.tight_layout()
outfname = f'plots/egfr-pl-per-residue-helix-{mut_type}.png'
fig.savefig(outfname, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved P-loop helix occupancy plot: {outfname}")


#write summary statistics
# Per-replica summary
output = open(f'activation-loop-turn-summary-{mut_type}.txt', 'w')
output.write('#system replica helix_turn_count helix_turn_count_SE short_helix_occupancy short_helix_occupancy_SE sheet_turn_count helix_fraction sheet_fraction n_frames\n')

for row in summary:
	output.write(f"{row['system']} {row['replica']} {row['helix_turn_count']:.4f} {row['helix_turn_count_se']:.4f} {row['short_helix_occupancy']:.4f} {row['short_helix_occupancy_se']:.4f} {row['sheet_turn_count']:.4f} {row['helix_fraction']:.4f} {row['sheet_fraction']:.4f} {row['n_frames']}\n")
output.close()
print(f"Wrote per-replica summary: activation-loop-turn-summary-{mut_type}.txt")

output = open(f'activation-loop-per-residue-helix-summary-{mut_type}.txt','w')
output.write('#system replica residue helix_occupancy sheet_occupancy\n')

for row in per_residue_summary:
	output.write(f"{row['system']} " f"{row['replica']} " f"{row['residue']} " f"{row['helix_occ']:.4f} " f"{row['sheet_occ']:.4f}\n")
output.close()
print(f"Wrote per-residue summary: "f"activation-loop-per-residue-helix-summary-{mut_type}.txt")


residue_occ = defaultdict(list)

for row in per_residue_summary:
	key = (row['system'], row['residue'])
	residue_occ[key].append(row['helix_occ'])
for system in sorted(set(x[0] for x in residue_occ.keys())):
	ranking = []
	for key, vals in residue_occ.items():
		sysname, residue = key
		if sysname == system:
			ranking.append((residue, np.mean(vals)))
	ranking.sort(key=lambda x: x[1],reverse=True)
	print(f"\n===== {system} Ranked AL residues =====")
	for rank, (residue, occ) in enumerate(ranking, start=1):
		print(f"{rank:2d}  {residue}  {occ:.3f}")





#Write A-loop and P-loop helicity-by-residue data
def write_helicity_plot_csv(per_residue_data,output_filename,loop_name):
	"""
	Write the per-replica helicity values and the system mean/SEM
	used in the helicity-versus-residue plot.
	One row = one system, replica, and residue.
	"""
	# Organize data as system -> replica -> residue -> helix occupancy

	occupancy_by_system_replica = defaultdict(lambda: defaultdict(dict))
	for row in per_residue_data:
		occupancy_by_system_replica[row['system']][row['replica']][row['residue']]= row['helix_occ']
	# Calculate the values used for the plotted mean and SE
	plot_statistics = {}
	for system, replica_data in occupancy_by_system_replica.items():
		# Collect every residue observed in at least one replica
		all_residues = sorted(set().union(*[set(replica_data[replica].keys())for replica in replica_data]))
		for residue in all_residues:
			#Include only replicas that contain this residue
			replica_values = np.array([replica_data[replica][residue]for replica in sorted(replica_data)if residue in replica_data[replica]])

			mean_occupancy = np.mean(replica_values)
			if len(replica_values) > 1:
				se_occupancy = (np.std(replica_values, ddof=1) /np.sqrt(len(replica_values)))
			else:
				se_occupancy = np.nan
			plot_statistics[(system, residue)] = {'mean': mean_occupancy,'se': se_occupancy,'n_replicas': len(replica_values)}


	# Write a helix  CSV
	with open(output_filename, 'w', newline='') as output:

		writer = csv.writer(output)
		writer.writerow(['loop', 'system', 'label', 'replica', 'residue', 'replica_helix_occupancy', 'replica_helix_occupancy_percent','system_mean_helix_occupancy', 'system_mean_helix_occupancy_percent', 'system_SE_helix_occupancy', 'system_SE_helix_occupancy_percent','n_replicas'])
		for row in sorted(per_residue_data,key=lambda x: (x['system'],x['replica'],x['residue'])):
			stats = plot_statistics[(row['system'], row['residue'])]
			writer.writerow([loop_name, row['system'], row['label'], row['replica'], row['residue'], f"{row['helix_occ']:.6f}",f"{100.0 * row['helix_occ']:.4f}", f"{stats['mean']:.6f}", f"{100.0 * stats['mean']:.4f}", f"{stats['se']:.6f}", f"{100.0 * stats['se']:.4f}",stats['n_replicas']])

	print(f"Wrote {loop_name} helicity plot spreadsheet: "f"{output_filename}")



# EDIT: Write spreadsheet-ready CSV for plotted A-loop sheet data
def write_sheet_plot_csv(per_residue_data,output_filename,loop_name):
	"""
	Write the per-replica helicity values and the system mean/SEM
	used in the sheet-occupancy-versus-residue plot. 
	One row = one system, replica, and residue.
	"""
	# Organize data as system -> replica -> residue -> helix occupancy

	occupancy_by_system_replica = defaultdict(lambda: defaultdict(dict))
	for row in per_residue_data:
		occupancy_by_system_replica[row['system']][row['replica']][row['residue']]= row['sheet_occ']
	# Calculate the values used for the plotted mean and SE
	plot_statistics = {}
	for system, replica_data in occupancy_by_system_replica.items():
		# Collect every residue observed in at least one replica
		all_residues = sorted(set().union(*[set(replica_data[replica].keys())for replica in replica_data]))
		for residue in all_residues:

			replica_values = np.array([replica_data[replica][residue]for replica in sorted(replica_data)if residue in replica_data[replica]])
			mean_occupancy = np.mean(replica_values)
			if len(replica_values) > 1:
				se_occupancy = (np.std(replica_values, ddof=1) /np.sqrt(len(replica_values)))
			else:
				se_occupancy = np.nan
			plot_statistics[(system, residue)] = {'mean': mean_occupancy,'se': se_occupancy,'n_replicas': len(replica_values)}

	#write sheet CSV
	with open(output_filename, 'w', newline='') as output:
		writer = csv.writer(output)
		writer.writerow(['loop', 'system', 'label', 'replica', 'residue', 'replica_sheet_occupancy', 'replica_sheet_occupancy_percent','system_mean_sheet_occupancy', 'system_mean_sheet_occupancy_percent', 'system_SE_sheet_occupancy', 'system_SE_sheet_occupancy_percent','n_replicas'])
		for row in sorted(per_residue_data,key=lambda x: (x['system'],x['replica'],x['residue'])):
			stats = plot_statistics[(row['system'], row['residue'])]
			writer.writerow([loop_name, row['system'], row['label'], row['replica'], row['residue'], f"{row['sheet_occ']:.6f}",f"{100.0 * row['helix_occ']:.4f}", f"{stats['mean']:.6f}", f"{100.0 * stats['mean']:.4f}", f"{stats['se']:.6f}", f"{100.0 * stats['se']:.4f}",stats['n_replicas']])
	print(f"Wrote {loop_name} sheet plot spreadsheet: "f"{output_filename}")


#A-loop spreadsheet used for the A-loop helicity plot
write_helicity_plot_csv(per_residue_summary,f'activation-loop-helicity-by-residue-{mut_type}.csv','A-loop')

#P-loop spreadsheet used for the P-loop helicity plot
write_helicity_plot_csv(pl_per_residue_summary,f'p-loop-helicity-by-residue-{mut_type}.csv','P-loop')


#A-loop spreadsheet used for the A-loop helicity plot
write_helicity_plot_csv(per_residue_summary,f'activation-loop-sheet-by-residue-{mut_type}.csv','A-loop')


#Write P-loop per-residue helicity table
output = open(f'p-loop-per-residue-helix-summary-{mut_type}.txt','w')
output.write('#system replica residue helix_occupancy\n')
for row in pl_per_residue_summary:
	output.write(f"{row['system']} " f"{row['replica']} " f"{row['residue']} " f"{row['helix_occ']:.4f}\n " )
output.close()
print(f"Wrote per-residue summary: "f"p-loop-per-residue-helix-summary-{mut_type}.txt")


# Mutant-level summary
mutant_summary = {}
for row in summary:
	mutant = row['system']
	if mutant not in mutant_summary:
		mutant_summary[mutant] = []
	mutant_summary[mutant].append({
		'helix_turn_count': row['helix_turn_count'],
		'short_helix_occupancy': row['short_helix_occupancy'],
		'helix_fraction': row['helix_fraction']
	})

output = open(f'activation-loop-turn-mutant-summary-{mut_type}.txt', 'w')
output.write('#system mean_helix_turn_count helix_turn_count_SE mean_short_helix_occupancy short_helix_occupancy_SE mean_helix_fraction\n')

for mutant in sorted(mutant_summary.keys()):
	helix_turn_values = np.array([x['helix_turn_count'] for x in mutant_summary[mutant]])
	helix_turn_mean = np.mean(helix_turn_values)
	if len(helix_turn_values) > 1:
		helix_turn_se = np.std(helix_turn_values, ddof=1) / np.sqrt(len(helix_turn_values))
	else:
		helix_turn_se = np.nan

	#Calculate 1-2 helix turn statistics for mutant
	short_helix_values = np.array([x['short_helix_occupancy'] for x in mutant_summary[mutant]])
	short_helix_mean = np.mean(short_helix_values)
	if len(short_helix_values) > 1:
		short_helix_se = np.std(short_helix_values, ddof=1) / np.sqrt(len(short_helix_values))
	else:
		short_helix_se = np.nan


	# Get helix fractions for this mutant
	helix_frac_values = np.array([x['helix_fraction'] for x in mutant_summary[mutant]])
	helix_frac_mean = np.mean(helix_frac_values)

	output.write(f'{mutant} {helix_turn_mean:.4f} {helix_turn_se:.4f} {short_helix_mean:.4f} {short_helix_se:.4f} {helix_frac_mean:.4f}\n')
output.close()
print(f"Wrote mutant-level summary: activation-loop-turn-mutant-summary-{mut_type}.txt")




