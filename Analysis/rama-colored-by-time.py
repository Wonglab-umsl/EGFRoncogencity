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



fig=plt.figure(constrained_layout=True,figsize=(nres,len(names)*nreplica))
#spec=gridspec.GridSpec(nrows=len(names),ncols=nres,figure=fig)
ticks=np.linspace(start=-180,stop=180,num=7)
blank_labels=['']*len(ticks)
for iname,name in enumerate(names):
	#load everything
	psffname='{0}.psf'.format(name)
	top=md.load_psf(psffname)

			
	for id in range(1,nreplica+1):
		dcdfname=f'{scratch}/{name}-{id}.dcd'
		print('reading trajectory ',dcdfname,flush=True)
		traj=md.load(dcdfname,top=top,stride=stride)

		#use mdtraj to calculate the phi and psi angles, convert from radians to degrees
		phi=md.compute_phi(traj)
		phi_atoms=phi[0]
		#print(iatom_by_res)
		phi=phi[1]*(180/np.pi)
		psi=md.compute_psi(traj)
		psi_atoms=psi[0]
		psi=psi[1]*(180/np.pi)
		print(phi.shape,psi.shape)

		#construct a dictionary relating residue numbers to indices
		phi_resnum_to_index={}
		psi_resnum_to_index={}
		resnames={}
		for ires,iatom in enumerate(phi_atoms[:,2]):
			residue=top.atom(iatom).residue.resSeq
			phi_resnum_to_index[residue]=ires
			resnames[residue]=top.atom(iatom).residue.name
		for ires,iatom in enumerate(psi_atoms[:,1]):
			residue=top.atom(iatom).residue.resSeq
			psi_resnum_to_index[residue]=ires
		#print(resnum_to_index)	
		#print(phi)
		#print(psi)
		#sys.exit(0)
		for res in range(res_start,res_end+1):
			if ((res not in phi_resnum_to_index) or (res not in psi_resnum_to_index)):
				continue
			phi_ires=phi_resnum_to_index[res]
			psi_ires=psi_resnum_to_index[res]
			resname=resnames[res]
			print("making plot for ",name,id,resname,res,flush=True)
			#print("first phi, psi: ",phi[0,phi_ires],psi[0,psi_ires])
			subplot_index=1+(res-res_start)+(iname*nreplica+id-1)*nres
			ax=fig.add_subplot(len(names)*nreplica,nres,subplot_index)
			ax.set_aspect('equal')
			#color by fraction of the trajectory
			time=np.linspace(start=0,stop=1,num=traj.n_frames)
			ax.scatter(phi[:,phi_ires],psi[:,psi_ires],marker=',',s=0.01,c=time,cmap='viridis')
			ax.set_xlim(left=-180,right=180)
			ax.set_ylim(bottom=-180,top=180)
			if (iname==len(names)-1):
				ax.set_xticks(ticks)
				ax.tick_params(axis='x',labelrotation=90)
			else:
				ax.set_xticks(ticks,blank_labels)
			if ((iname==0) and (id==1)):
				ax.set_title('{0} {1:d}'.format(resname,res),fontdict={'fontsize':8})
			if (res==res_start):
				ax.set_yticks(ticks)
				ax.set_ylabel(f'{labels[name]} {id:d}',fontsize=5)
				#ax.set_ylabel(f'{labels[name]} rep. {id:d}',fontsize=6)
			else:
				ax.set_yticks(ticks,blank_labels)
			ax.tick_params(labelsize=8)
			ax.grid()


#finish and save the plot
fig.subplots_adjust(wspace=0,hspace=0)
outfname=f'plots/egfr-rama-{mut_type}-{res_start:d}-{res_end:d}-bytime.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.tight_layout()
plt.close(fig)

