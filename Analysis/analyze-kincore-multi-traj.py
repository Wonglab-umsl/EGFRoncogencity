#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pdb
import pandas as pd
#nsection=int(sys.argv[1])
#nframes=int(sys.argv[2])
nreplica=int(sys.argv[1])
#read the list of names
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



#dt=0.001 #frame interval in ns
#we might set up counts here
n_active=np.zeros((len(names),nreplica),dtype=np.int32)
n_inactive=np.zeros((len(names),nreplica),dtype=np.int32)



#these two lists are for plotting
for iname,name in enumerate(names):
	for id in range(1,nreplica+1):
		fname=f'{name}-{id}-kincore-output'.format(name)
		try:
			input=open(fname,'r')
		except IOError:
			continue
		print('reading file ',fname,flush=True)
		for line in input:
			words=line.split()
			#we only need to process those lines that contain Labels
			if (words[5]=='Labels'):
				#words[0] is the frame number!
				iframe=int(words[0])
				#determine which section we are in
				#isection=int(nsection*(iframe-1)/nframes)
				#they were done at 100 frame intervals; the "sections" are staggered 
				#isection=(iframe//100)%nsection
				#your turn -- process this line and increment counters as needed
				if (words[12]=='Active'):
					n_active[iname,id-1]+=1
				elif (words[12]=='Inactive'):
					n_inactive[iname,id-1]+=1
				else:
					print('error',line)
					sys.exit(-1)
		input.close()

frac_active=n_active/(n_active+n_inactive)
#frac_active[:,0] is the fraction active for all mutants in the 1st trajectory, frac_active[:,1] the 2nd, etc.

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(frac_active[:,0],frac_active[:,1],marker='+')
for iname,name in enumerate(names):
	ax.annotate(labels[name],(frac_active[iname,0],frac_active[iname,1]),fontsize=5)
ax.set_aspect('equal')
ax.axline((0,0),(1,1),transform=ax.transAxes)
ax.set_xlim(left=0,right=1)
ax.set_ylim(bottom=0,top=1)
ax.set_xlabel('fraction active in trajectory 1')
ax.set_ylabel('fraction active in trajectory 2')
outfname=f'plots/egfr-frac-active-comparison.png'
fig.savefig(outfname,format='png',bbox_inches='tight',dpi=600)
plt.close(fig)
#pdb.set_trace()
#print(n_active,' active frames and ',n_inactive,' inactive frames',flush=True)
# We don't need to print Salt_br-out counts_
#print out the information in counts_by_label about what the labels are and how many of each there are
#print(counts_by_label)
#or:
#for dunbrack_label in counts_by_label.keys():
	#print(dunbrack_label,counts_by_label[dunbrack_label])


#print(BLAminus,' frames with BLAminus and ',BLAminus=1,' frames with BLAminus+1',flush=True) 
#output the information contained in the counters to files as appropriate
#pdb.set_trace()
expanded_names=[f'{name}-{id}' for name in names for id in range(1,nreplica+1)]
active_df=pd.DataFrame(data=frac_active.reshape(len(names)*nreplica,),index=expanded_names,columns=['Active'])

active_df.to_csv('relative-population-active-multi-traj.csv')

#rmsd_df.to_csv('rmsf-by-section.csv')



