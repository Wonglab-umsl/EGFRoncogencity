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
import pickle

nreplica=int(sys.argv[1])
stride=1
if (len(sys.argv)>2):
	stride=int(sys.argv[2])
#time between frames
dt=0.001*stride
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


scratch='/cluster/pixstor/wongch-lab/Victor/egfr/dcd-joined'
#we will add the names of the mutants later
#names=['egfr-wt-active-charmm','egfr-wt-inactive-charmm','egfr-t747t751p-active-charmm','egfr-t751i759n-active-charmm'] 
ss={}
for name in names:
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)
	#reffname='data/{0}-solvated.pdb'.format(name)
	#reftraj=md.load(reffname)
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id:d}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)
		print(f'total {traj.n_frames:d} frames read',flush=True)
		#read the residue numbers
		ca_atoms=top.select('name CA')
		resnums=[]
		for idx in ca_atoms:
			resnums.append(top.atom(idx).residue.resSeq)
	
		ss[(name,id)]=md.compute_dssp(traj,simplified=True)
		del traj

#store the secondary structure info as a pickle file for later plotting
output=open('/cluster/pixstor/wongch-lab/Victor/egfr/egfr-sec-struc-info.p','wb')
pickle.dump(ss,output)
output.close()

