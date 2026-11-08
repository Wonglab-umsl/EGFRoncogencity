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

#start from 696 = 1
#721-722 is missing
#746-750 is deleted , 
class MyModel(automodel):
   def select_atoms(self):
      return selection(self.residue_range('25:A', '26:A'), self.residue_range('49:A','55:A'))



a=MyModel(env, alnfile=alnfile, knowns=known, sequence=target)
a.starting_model=1
a.ending_model=1
a.make()


