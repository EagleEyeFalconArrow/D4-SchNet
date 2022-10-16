# In this file, we will generate the neighbouring list for the atoms in the system. This is a pre-processing step for the SchNet model.

import argparse

from schnet.data import generate_neighbor_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('srcpath', help='path to dataset in ase.db format')
    parser.add_argument('output', help='output file')
    parser.add_argument('cutoff', help='distance cutoff', type=float)
    args = parser.parse_args()

    output = args.output
    if not output[-3:] == '.db':
        output += '.db'

    generate_neighbor_dataset(args.srcpath, output, args.cutoff)
