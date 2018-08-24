# This module is mostly borrowed from Olivecrona original implementation 
# (https://github.com/MarcusOlivecrona/REINVENT).
# We adjusted some functions for this project.
#

import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=150):
        self.special_tokens = ['END', 'START']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars)))) 
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[C@@H]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['END']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('END')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}


    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line.split()[0])

    def __getitem__(self, i): 
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[C@H]' as one token."""
    regex = '(\[[^\[\]]{1,10}\])' 
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('END')
    return tokenized

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of isomericSMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)          
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol,isomericSmiles=True))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


def filter_mol(mol, max_heavy_atoms=100, min_heavy_atoms=10, element_list=[6,7,8,9,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False


def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
#         print(tokenized)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    print('Filtered library size: %d'%len(smiles_list_valid))
    return smiles_list_valid


def filter_file_on_chars(smiles_list, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    chars = []
    for line in smiles_list:
        smiles.append(line.split()[0])
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print('Vocabulary size: %d'%len(chars))
    print('Origin library size: %d'%len(smiles))
    valid_smiles = filter_on_chars(smiles, chars)
    with open("tl_filtered.smi", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def construct_vocabulary(smiles_list,fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)

            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)
