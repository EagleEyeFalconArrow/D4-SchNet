# In this file, given the path to an MD model, we print the energy and forces of the system (for the desired inputs). 
# Also, depending on the input, we can choose to relax the positions of the molecules in the atomistic system as well.


import os
import argparse
from ase.io import read, write
from schnet import md

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to MD models')
    parser.add_argument('input', help='xyz input file')
    parser.add_argument('--relax', help='relax molecule', action='store_true')
    args = parser.parse_args()

    force_path = os.path.join(args.model_path, 'force_model')
    energy_path = os.path.join(args.model_path, 'energy_model')
    if not os.path.exists(energy_path):
        energy_path = force_path

    at = read(args.input)

    mdpred = md.SchNetMD(energy_path, force_path, nuclear_charges=at.numbers)
    energy, forces = mdpred.get_energy_and_forces(at.positions)
    print('Energy:', energy)
    print('Force:', forces)

    if args.relax:
        eq_pos = mdpred.relax(at.positions, eps=1e-4, rate=5e-4)
        at.set_positions(eq_pos)
        RELOUT = '.'.join(args.input.split('.')[:-1])+'_relaxed.xyz'
        write(RELOUT, at, format='xyz')
        print('Relaxed:', eq_pos)
        