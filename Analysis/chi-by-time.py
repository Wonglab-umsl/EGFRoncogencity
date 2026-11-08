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


#res_list=[854,856]
#nres=len(res_list)
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



fig=plt.figure(constrained_layout=False,figsize=(nres,len(names)*nreplica))
gs=gridspec.GridSpec(nrows=len(names)*nreplica,ncols=nres+1,figure=fig)

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

		chi1_atoms, chi1=md.compute_chi1(traj)
		#pdb.set_trace()
		#construct a dictionary relating residue numbers to indices
		chi1_resnum_to_index={}
		resnames={}
		for ires,iatom in enumerate(chi1_atoms[:,0]):
			residue=top.atom(iatom).residue.resSeq
			chi1_resnum_to_index[residue]=ires
			resnames[residue]=top.atom(iatom).residue.name
		#print(resnum_to_index)	
		#print(phi)
		#print(psi)
		#sys.exit(0)
		for res in range(res_start,res_end+1):
			if ((res not in chi1_resnum_to_index)):
				continue
			chi1_ires=chi1_resnum_to_index[res]
			resname=resnames[res]
			print("making plot for ",name,id,resname,res,flush=True)
			#print("first phi, psi: ",phi[0,phi_ires],psi[0,psi_ires])
			#subplot_index=1+(res-res_start)+(iname*nreplica+id-1)*nres
			#ax=fig.add_subplot(len(names)*nreplica,nres,subplot_index)
			ax=fig.add_subplot(gs[iname*nreplica+id-1,res-res_start],projection='polar')
			ax.set_aspect('equal')
			#color by fraction of the trajectory
			#time=np.linspace(start=0,stop=1,num=traj.n_frames)
			time=np.linspace(start=dt*stride,stop=traj.n_frames*dt*stride,num=traj.n_frames)
			#this is hard coded -- the longest trajectory will be 1 ns
			ax.scatter(chi1[:,chi1_ires],time,marker=',',c='k',s=0.1,lw=0.1)
			ax.set_rmin(0)
			ax.set_rmax(1000)
			ax.grid(True,lw=0.1)

			if ((iname==0) and (id==1)):
				ax.set_title('{0} {1:d}'.format(resname,res),fontdict={'fontsize':8})
			if (res==res_start):
				ax.set_ylabel(f'{labels[name]}-{id:d}',fontsize=5)
			ax.tick_params(labelsize=2)
			ax.tick_params(pad=-8.0,axis='x')
			#angles_rad = np.linspace(0, 2*np.pi, 9, endpoint=False)
			#ax.set_thetagrids(angles=angles_rad,frac=1.1)



#finish and save the plot
fig.subplots_adjust(wspace=0.3,hspace=0)
outfname=f'plots/egfr-chi1-{mut_type}-{res_start:d}-{res_end:d}-bytime.png'
#plt.tight_layout()
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)

