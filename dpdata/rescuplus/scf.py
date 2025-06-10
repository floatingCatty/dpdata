import json
import numpy as np

Hartree2eV = 27.21138602
Bohr2Angstrom = 0.52917721067


def get_frame(f):

    data = {
        "atom_names": [],
        "atom_numbs": [],
        "atom_types": [],
        "cells": np.array([]),
        "coords": np.array([]),
        "energies": np.array([]),
        "forces": np.array([]),
    }

    with open('data.json', 'r') as file:
        f = json.load(file)
        atoms = f["system"]["atoms"]
        cells = f["system"]["cell"]

        pos = np.array(atoms["positions"]["array"]).reshape(-1,3) # bohr
        cell = cells["avec"]["array"].reshape(3,3) # bohr

        # get energy
        energy = f["energy"]["etot"]["magnitude"].reshape(-1) # hartree
        # get force
        force = f["energy"]["forces"]["array"].reshape(-1,3) # hartree / bohr
        # get stress
        stress = f["energy"]["stress"]["array"].reshape(3,3) # hartree / bohr ** 3
    
        data["cells"] = cell.reshape(1,3,3) * Bohr2Angstrom
        data["coords"] = pos.reshape(1,-1,3) * Bohr2Angstrom
        data["energy"] = energy * Hartree2eV
        data["forces"] = force[None,:,:] * (Hartree2eV / Bohr2Angstrom)
        data["orig"] = np.zeros(3)

        cell = data["cells"][0]
        stress *= Hartree2eV / Bohr2Angstrom**3
        stress *= np.abs(np.linalg.det(cell))
        data["virials"] = stress[np.newaxis, :, :]
    

    return data

