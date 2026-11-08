#!/usr/bin/env python

import sys
#import modeller

alnfile=sys.argv[1]
known=sys.argv[2]
target=sys.argv[3]
print(alnfile,known,target)
# Comparative modeling by the automodel class
from modeller import * # Load standard Modeller classes
from modeller.automodel import * # Load the automodel class

#log.verbose() # request verbose output
env = Environ() # create a new MODELLER environment to build this model in

# directories for input atom files
env.io.atom_files_directory = ['.']

a=AutoModel(env, alnfile=alnfile, knowns=known, sequence=target)
a.starting_model=1
a.ending_model=1
a.make()


