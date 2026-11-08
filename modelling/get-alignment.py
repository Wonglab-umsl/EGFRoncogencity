#!/usr/bin/env python

import sys


aadict={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HSD':'H','HSE':'H','HSP':'H','HIS':'H','ILE':'I','LYS':'K',
        'LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

#read the sequence from a pdb file and create a dictionary, also identify starting and ending residues 
def read_sequence(pdbfname,chain):
	start=1000000
	end=-1
	d={}
	with open(pdbfname,'r') as input:
		for line in input:
			if (line[0:4]!='ATOM'):
				continue
			if (line[12:16].replace(' ','')!="CA"):
				continue
			if (line[21]!=chain):
				continue
			res=int(line[22:26])
			rname=line[17:20]
			if (rname not in aadict):
				print("unrecognized amino acid {0} at residue {1:d}".format(rname,res))
				continue
			d[res]=aadict[rname]
			start=min(start,res)
			end=max(end,res)
	input.close()
	return start,end,d


pdbfname=sys.argv[1]
chain=sys.argv[2]
start,end,d=read_sequence(pdbfname,chain)
if (len(sys.argv)>3):
	start=int(sys.argv[3])
	end=int(sys.argv[4])


#print(d)
outseq=''
for res in range(start,end+1):
	if (res in d):
		outseq=outseq+d[res]
	else:
		outseq=outseq+'-'
print(start,end)
print(outseq)
