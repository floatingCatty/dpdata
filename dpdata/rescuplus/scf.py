import json
import numpy as np
import os


Hartree2eV = 27.21138602
Bohr2Angstrom = 0.52917721067

    
def compress_formula(formula):
    sform = split_formula(formula)
    cform = []
    tmp0 = sform[0]
    count = 0
    for s in sform:
        if s == tmp0:
            count += 1
        else:
            if count > 1:
                cform.append(f"{tmp0}({count})")
            else:
                cform.append(f"{tmp0}")
            count = 1
            tmp0 = s
    if count > 1:
        cform.append(f"{tmp0}({count})")
    else:
        cform.append(f"{tmp0}")
    return "".join(cform)


def split_formula(formula):
    sform = []
    i = 0
    while i < len(formula):
        tmp = formula[i]
        i += 1
        while i < len(formula) and not formula[i].isupper():
            tmp += formula[i]
            i += 1
        sform.append(tmp)
    return sform

def get_formula(formula):
    if "(" in formula: # compressed format like Ga(16)N(16)
        lists = formula.split("(")
        split_lists = []
        for s in lists:
            if ")" in s:
                ss = s.split(")")
            else:
                ss = [s]
            
            split_lists.extend(ss)
        if len(split_lists[-1]) == 0:
            split_lists = split_lists[:-1]
        
        split_lists_out = []
        for sym in split_lists:
            if not sym.isdigit():
                split_lists_out += split_formula(sym)
            else:
                num = int(sym)
                if num > 1:
                    split_lists_out += [split_lists_out[-1]] * (int(sym)-1)
                else:
                    raise ValueError("Invalid formula: the number of compressed element format need to be graater than 0.")
    else:
        split_lists_out = split_formula(formula)

    

    return split_lists_out

def get_frame(f):
    f = os.path.join(f, "nano_scf_out.json")
    data = {
        "atom_names": [],
        "atom_numbs": [],
        "atom_types": [],
        "cells": np.array([]),
        "coords": np.array([]),
        "energies": np.array([]),
        "forces": np.array([]),
    }

    with open(f, 'r') as file:
        f = json.load(file)
        atoms = f["system"]["atoms"]
        cells = f["system"]["cell"]

        formula = atoms["formula"]
        formula = get_formula(formula)

        out = []
        out.append(formula[0])
        number = [0] * len(set(formula))
        number[0] = 1
        count = 0
        for sym in formula[1:]:
            if sym == out[-1]:
                number[count] += 1
                continue
            else:
                count += 1
                number[count] += 1
                out.append(sym)


        pos = np.array(atoms["positions"]["magnitude"]["array"]).reshape(-1,3) # bohr
        cell = np.array(cells["avec"]["magnitude"]["array"]).reshape(3,3) # bohr

        # get energy
        energy = np.array(f["energy"]["etot"]["magnitude"]).reshape(-1) # hartree
        # get force
        if f["energy"]["forces_return"]:
            force = np.array(f["energy"]["forces"]["magnitude"]["array"]).reshape(-1,3) # hartree / bohr
        else:
            force = None
        # get stress
        if f["energy"]["stress_return"] == True:
            stress = np.array(f["energy"]["stress"]["magnitude"]["array"]).reshape(3,3) # hartree / bohr ** 3
        else:
            stress = None

        data["atom_names"] = formula
        data["atom_numbs"] = number
        data["atom_types"] = np.array([i for i in range(len(number)) for j in range(number[i])])
        data["cells"] = cell.reshape(1,3,3) * Bohr2Angstrom
        data["coords"] = pos.reshape(1,-1,3) * Bohr2Angstrom
        data["energies"] = energy * Hartree2eV
        if force is not None:
            data["forces"] = force[None,:,:] * (Hartree2eV / Bohr2Angstrom)
        data["orig"] = np.zeros(3)

        cell = data["cells"][0]

        if stress is not None:
            stress *= Hartree2eV / Bohr2Angstrom**3
            stress *= np.abs(np.linalg.det(cell))
        
            data["virials"] = stress[np.newaxis, :, :]
    
    return data

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z