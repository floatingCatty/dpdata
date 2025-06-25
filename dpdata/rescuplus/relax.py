import json
import numpy as np
import os
from ase.io import Trajectory

Hartree2eV = 27.21138602
Bohr2Angstrom = 0.52917721067

def CheckFile(ifile):
    if not os.path.isfile(ifile):
        print(f"Can not find file {ifile}")
        return False
    return True

def get_traj_path(f):
    traj_path = os.path.join(f, "nano_rlx.traj")
    return traj_path

def get_log_path(f):
    log_path = os.path.join(f, "resculog.out")
    return log_path

def get_frame(f):
    traj_path = get_traj_path(f)
    traj = Trajectory(traj_path)

    formula = traj[0].get_chemical_symbols()
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

    data = {
        "atom_names": out,
        "atom_numbs": number,
        "atom_types": np.array([i for i in range(len(number)) for j in range(number[i])]),
        "cells": np.array([]),
        "coords": np.array([]),
        "energies": np.array([]),
        "forces": np.array([]),
    }

    poss = []
    cells = []
    energys = []
    forces = []
    stresses = []

    for atoms in traj:
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        energy = atoms.get_total_energy()
        force = atoms.get_forces()
        stress = atoms.get_stress(voigt=False) if hasattr(atoms, 'get_stress') else None

        if stress is not None:
            stress *= Hartree2eV / Bohr2Angstrom**3
            stress *= np.abs(np.linalg.det(cell))

        poss.append(pos)
        cells.append(cell)
        energys.append(energy)
        forces.append(force)
        stresses.append(stress)

    data["cells"] = np.stack(cells) * Bohr2Angstrom
    data["coords"] = np.stack(poss) * Bohr2Angstrom
    data["energies"] = np.stack(energys) * Hartree2eV
    data["forces"] = np.stack(forces) * (Hartree2eV / Bohr2Angstrom)
    data["orig"] = np.zeros(3)

    if stress is not None:
        data["virials"] = np.array(stresses)
    

    return data

