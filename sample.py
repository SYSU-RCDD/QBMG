#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
import data_struct as ds
from data_struct import MolData, Vocabulary
from model import RNN
import sys


def Sample(filename, enumerate_number):
    voc = Vocabulary(init_from_file="./Voc")  
    Prior = RNN(voc)
    print(filename, enumerate_number)
    # Can restore from a saved RNN
    Prior.rnn.load_state_dict(torch.load(filename))
    totalsmiles = set()
    enumerate_number = int(enumerate_number)
    molecules_total = 0
    for epoch in range(1, 10000):
        seqs, likelihood, _ = Prior.sample(100)
        valid = 0      
        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            if Chem.MolFromSmiles(smile):
                valid += 1
                totalsmiles.add(smile)
                       
        molecules_total = len(totalsmiles)
        print(("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs))))
        print(valid, molecules_total, epoch)
        if molecules_total > enumerate_number:
            break
    return totalsmiles

if __name__ == "__main__":
    filename = sys.argv[1]
    n = sys.argv[2]
    print(filename)
    totalsmiles=Sample(filename,n)
    f = open('./sample.smi', 'w')  
    for smile in totalsmiles:
        f.write(smile + "\n")
    f.close()
    print('Sampling completed')
