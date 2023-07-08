#!/usr/bin/env python
# Authors: Nike Dattani (nike@hpqc.org) and Vandan Revanur (vandan@hpqc.org)

import requests  # First run "pip install requests"
import pyscf
from pyscf.tools.get_xyz import get_xyz

geometry_database =   'https://raw.githubusercontent.com/HPQC-LABS/goDatabase/master/benchmark_sets/gw5000.txt'

molecule_name =       'C10H15NO2S2'
geometry_identifier = 'GW5000'
geometry_version =    '0'
geometry_name = molecule_name+':'+geometry_identifier+'.v'+geometry_version

page = requests.get(geometry_database)
data = page.text
molecule_xyz = get_xyz(data, geometry_name)

print(molecule_xyz)
mol = pyscf.M(atom=molecule_xyz, basis='sto-3g', verbose=5, spin=0, charge=0)
print(mol)

