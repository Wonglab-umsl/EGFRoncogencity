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
#from itertools import combinations
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


#DSSP secondary-structure states of interest
Helix_states={'H', 'G'}   #alpha-helix + 3_10 helix
Turn_state = 'T'


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


	print(f"{name}: AL residues {AL_START}-{AL_END} map to array indices {[residue_to_index[name].get(r, 'N/A') for r in range(AL_START, AL_END+1)]}")
	break  # PSF topology should be same for all replicates, use first name



#Analyze turn presence in the activation loop
def detect_short_helical_segment(ss_sequence, min_len=3, max_len=4,helix_states=('H','G')):
	"""
	Detect 1-2 residue helical segments consisting of
	alpha helices (H) and/or 3_10 helices (G)


	Examples counted: H HH G GG HG GH 

	"""

	run_length=0

	for ss in ss_sequence:
		if ss in helix_states:
			run_length +=1
		else:
			if min_len <= run_length <= max_len:
				#Check that helix is flanked by non-helix (not at boundaries)
				#if i - helix_count > 0 and i < len(ss_sequence):
				return True
			run_length=0

	#check final helix block
	if min_len <= run_length <= max_len:
		return True

	return False


def detect_short_turn(ss_sequence, min_len=3, max_len=4):
	"""
	Detect 1-2 residue DSSP turn blocks (T)
	"""
	run_length=0

	for ss in ss_sequence:
		if ss=='T':
			run_length +=1
		else:
			if min_len <= run_length <= max_len:
				return True
			run_length=0

	if min_len <= run_length <= max_len:
		return True

	return False


#def calculate_helix_content(ss_sequence):
#	"""Calculate fraction of helical residues"""
#	if len(ss_sequence) == 0:
#		return np.nan
#	return np.sum(np.array(ss_sequence) == 'H') / len(ss_sequence)

#Extract AL residue indices for each system
def get_al_indices(name, psf_top, AL_START, AL_END):
	"""Get array indices for AL residues"""
	ca_atoms = psf_top.select('name CA')
	resnums = []
	for idx in ca_atoms:
		resnums.append(psf_top.residue(psf_top.atom(idx).residue.index).resSeq)

	al_indices = []
	for resnum in range(AL_START, AL_END + 1):
		if resnum in resnums:
			al_indices.append(resnums.index(resnum))

	return al_indices

summary = []
turn_presence_data = []
turn_presence_labels = []

for iname, name in enumerate(names):
	print(f"\nProcessing system: {name}")

	#Load PSF for this system
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	al_indices = get_al_indices(name, top, AL_START, AL_END)

	print(f"AL residues {AL_START}-{AL_END} correspond to array indices: {al_indices}")

	for id in range(1, nreplica + 1):
		print(f"  Analyzing replica {id}...")
		# Get secondary structure for this trajectory
		if (name, id) not in ss:
			print(f"    Warning: No SS data for {name}-{id}")
			continue
		ss_data = ss[(name, id)][::stride]  # Apply stride if needed
		unique_ss = np.unique(ss_data)
		print("Unique DSSP states:", unique_ss)
		nframes = ss_data.shape[0]

		print(f" Total frames: {nframes}")

		# Extract AL secondary structure
		al_ss = ss_data[:, al_indices]  # (n_frames, n_al_residues)
		print("Total H residues in AL:",np.sum(al_ss == 'H'))
		#pdb.set_trace()

		# Calculate metrics for each frame
		nturn_helix = []
		nturn_sheet = []
		nhelix = []
		nsheet = []
		nturn_1to2_helix = []
		nturn_1to2_turn = []
		nturn_dssp = []
		nhelix_hg = []


		for frame_idx in range(nframes):

			ss_frame = al_ss[frame_idx, :]

			if np.any(ss_frame == 'H'):
				print("Frame:", frame_idx)
				print(''.join(ss_frame))

			nturn_helix.append(count_consecutive(ss_frame,'H'))
			nturn_sheet.append(count_consecutive(ss_frame,'E'))

			# Calculate fractions
			nhelix.append(np.sum(ss_frame == 'H') / len(ss_frame))
			nsheet.append(np.sum(ss_frame == 'E') / len(ss_frame))

			#detect 1-2 helix turn
			has_short_helix = detect_short_helical_segment(ss_frame, min_len=1,max_len=2,helix_states=('H','G'))
			nturn_1to2_helix.append(1 if has_short_helix else 0)

			has_short_turn = detect_short_turn(ss_frame, min_len=1,max_len=2)
			nturn_1to2_turn.append(1 if has_short_turn else 0)

			#fraction of H+G residues
			nhelix_hg.append(np.sum(np.isin(ss_frame, ['H','G']))/len(ss_frame))
			#DSSP turn present
			nturn_dssp.append(1 if np.any(ss_frame == 'T') else 0)




		nturn_helix = np.array(nturn_helix)
		nturn_sheet = np.array(nturn_sheet)
		nhelix = np.array(nhelix)
		nsheet = np.array(nsheet)
		nturn_1to2_helix = np.array(nturn_1to2_helix)
		nturn_1to2_turn = np.array(nturn_1to2_turn)
		nturn_dssp = np.array(nturn_dssp)
		nhelix_hg = np.array(nhelix_hg)



		# Calculate summary statistics
		helix_turn_occupancy = np.mean(nturn_helix)
		sheet_turn_occupancy = np.mean(nturn_sheet)
		helix_fraction_occupancy = np.mean(nhelix)
		sheet_fraction_occupancy = np.mean(nsheet)
		helix_turn_occupancy_se = np.std(nturn_helix) / np.sqrt(len(nturn_helix))
		helix_1to2_turn_occupancy = np.mean(nturn_1to2_helix)
		dssp_turn_occupancy = np.mean(nturn_dssp)
		short_turn_occupancy = np.mean(nturn_1to2_turn)
		hg_helix_fraction = np.mean(nhelix_hg)
		helix_1to2_turn_occupancy_se = np.std(nturn_1to2_helix) / np.sqrt(len(nturn_1to2_helix))

		print("Mean helix fraction =", np.mean(nhelix))
		print("Mean HG fraction    =", np.mean(nhelix_hg))
		print("1-2 helix occupancy =", np.mean(nturn_1to2_helix))


		summary.append({
			'system': name,
			'replica': id,
			'helix_turn_count': helix_turn_occupancy,
			'helix_turn_count_se': helix_turn_occupancy_se,
			'sheet_turn_count': sheet_turn_occupancy,
			'helix_fraction': helix_fraction_occupancy,
			'helix_1to2_turn_occupancy': helix_1to2_turn_occupancy,
			'helix_1to2_turn_occupancy_se': helix_1to2_turn_occupancy_se,
			'dssp_turn_occupancy': dssp_turn_occupancy,
			'short_turn_occupancy': short_turn_occupancy,
			'hg_helix_fraction': hg_helix_fraction,
			'sheet_fraction': sheet_fraction_occupancy,
			'n_frames': nframes
		})
		print(summary[-1])

		print(f"{name}-{id}: H/G 1-2 helices={helix_1to2_turn_occupancy:.3f} DSSP turns={dssp_turn_occupancy:.3f} 1-2 DSSP turns={short_turn_occupancy:.3f} H+G fraction={hg_helix_fraction:.3f}")

		turn_presence_data.append(nturn_helix)
		turn_presence_labels.append(f'{labels[name]} rep. {id:d}')
		turn_presence_data.append(nturn_1to2_helix)
		turn_presence_labels.append(f'{labels[name]} rep. {id:d} (1-2 helix turns)')


#Generate time-course plot of turn presence
if len(turn_presence_data) > 0:
	fig = plt.figure(figsize=(14, len(turn_presence_data) * 1.2))

	for idx, (turn_data, label) in enumerate(zip(turn_presence_data, turn_presence_labels)):
		nframes = len(turn_data)
		frames = np.arange(nframes)
		time_ns = frames * dt

		# Calculate rolling average for smoothing
		window_size = max(5, nframes // 50)
		if window_size < len(turn_data):
			rolling_turn = np.convolve(turn_data, np.ones(window_size) / window_size, mode='same')
		else:
			rolling_turn = turn_data.copy()
		#print(f"Window size: {window_size}, turn_data shape: {turn_data.shape}, rolling_turn shape: {rolling_turn.shape}, time_ns shape: {time_ns.shape}")


		ax = fig.add_subplot(len(turn_presence_data), 1, idx + 1)
		ax.fill_between(time_ns, 0, turn_data, alpha=0.3, color='steelblue', label='Helical turn blocks')
		ax.plot(time_ns, rolling_turn, color='darkblue', linewidth=1.5, label='Rolling average')
		ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
		ax.set_xlim(left=0, right=time_ns[-1])
		ax.set_ylim(bottom=-0.05, top=1.05)
		ax.set_ylabel(label, fontsize=8)
		ax.tick_params(labelsize=7)
		ax.grid(alpha=0.3)

		if idx == len(turn_presence_data) - 1:
			ax.set_xlabel('time (ns)', fontsize=8)
		if idx == 0:
			ax.set_title('Activation-Loop Helical Turn Block Over Time', fontdict={'fontsize': 12})

	fig.tight_layout()
	outfname = f'plots/egfr-al-turn-timecourse-{mut_type}.pdf'  #png
	fig.savefig(outfname, format='png', bbox_inches='tight')   #dpi=600
	plt.close(fig)
	print(f"Saved time-course plot: {outfname}")


#generate summary heatmap
if len(turn_presence_data) > 0:
	# Pad data to same length
	max_frames = max([len(d) for d in turn_presence_data])
	turn_heatmap = []
	for data in turn_presence_data:
		data_float=data.astype(np.float32)
		padded = np.pad(data_float, (0, max_frames - len(data_float)), mode='constant', constant_values=0)
		turn_heatmap.append(padded)

	turn_heatmap = np.array(turn_heatmap)

	fig = plt.figure(figsize=(12, max(4, len(turn_presence_data) * 0.3)))
	ax = fig.add_subplot(111)
	im = ax.imshow(turn_heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
	ax.set_yticks(np.arange(len(turn_presence_labels)))
	ax.set_yticklabels(turn_presence_labels, fontsize=8)
	ax.set_xlabel('Frame index', fontsize=10)
	ax.set_ylabel('System and Replica', fontsize=10)

	cbar = plt.colorbar(im)
	cbar.set_label('Turn present (1=yes, 0=no)', fontsize=10)
	ax.set_title('Activation-Loop Helical Turn Blocks Presence Heatmap', fontsize=12)

	fig.tight_layout()
	outfname = f'plots/egfr-al-turn-heatmap-{mut_type}.png'
	fig.savefig(outfname, format='png', bbox_inches='tight', dpi=600)
	plt.close(fig)
	print(f"Saved heatmap: {outfname}")

#write summary statistics
# Per-replica summary
output = open(f'activation-loop-turn-summary-{mut_type}.txt', 'w')
output.write('#system replica helix_turn_count helix_turn_count_SE helix_1to2_turn_occupancy helix_1to2_turn_occupancy_SE dssp_turn_occupancy short_turn_occupancy hg_helix_fraction sheet_turn_count helix_fraction sheet_fraction n_frames\n')

for row in summary:
	output.write(f"{row['system']} {row['replica']} {row['helix_turn_count']:.4f} {row['helix_turn_count_se']:.4f} {row['helix_1to2_turn_occupancy']:.4f} {row['helix_1to2_turn_occupancy_se']:.4f} {row['dssp_turn_occupancy']:.4f} {row['short_turn_occupancy']:.4f} {row['hg_helix_fraction']:.4f}  {row['sheet_turn_count']:.4f} {row['helix_fraction']:.4f} {row['sheet_fraction']:.4f} {row['n_frames']}\n")
output.close()
print(f"Wrote per-replica summary: activation-loop-turn-summary-{mut_type}.txt")

# Mutant-level summary
mutant_summary = {}
for row in summary:
	mutant = row['system']
	if mutant not in mutant_summary:
		mutant_summary[mutant] = []
	mutant_summary[mutant].append({
		'helix_turn_count': row['helix_turn_count'],
		'helix_1to2_turn_occupancy': row['helix_1to2_turn_occupancy'],
		'helix_fraction': row['helix_fraction']
	})

output = open(f'activation-loop-turn-mutant-summary-{mut_type}.txt', 'w')
output.write('#system mean_helix_turn_count helix_turn_count_SE mean_helix_1to2_turn_occupancy helix_1to2_turn_occupancy_SE mean_helix_fraction\n')

for mutant in sorted(mutant_summary.keys()):
	helix_turn_values = np.array([x['helix_turn_count'] for x in mutant_summary[mutant]])
	helix_turn_mean = np.mean(helix_turn_values)
	if len(helix_turn_values) > 1:
		helix_turn_se = np.std(helix_turn_values, ddof=1) / np.sqrt(len(helix_turn_values))
	else:
		helix_turn_se = np.nan

	#Calculate 1-2 helix turn statistics for mutant
	helix_1to2_turn_values = np.array([x['helix_1to2_turn_occupancy'] for x in mutant_summary[mutant]])
	helix_1to2_turn_mean = np.mean(helix_1to2_turn_values)
	if len(helix_1to2_turn_values) > 1:
		helix_1to2_turn_se = np.std(helix_1to2_turn_values, ddof=1) / np.sqrt(len(helix_1to2_turn_values))
	else:
		helix_1to2_turn_se = np.nan


	# Get helix fractions for this mutant
	helix_frac_values = np.array([x['helix_fraction'] for x in mutant_summary[mutant]])
	helix_frac_mean = np.mean(helix_frac_values)

	output.write(f'{mutant} {helix_turn_mean:.4f} {helix_turn_se:.4f} {helix_1to2_turn_mean:.4f} {helix_1to2_turn_se:.4f} {helix_frac_mean:.4f}\n')
output.close()
print(f"Wrote mutant-level summary: activation-loop-turn-mutant-summary-{mut_type}.txt")




