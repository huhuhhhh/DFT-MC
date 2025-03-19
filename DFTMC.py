# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation for High Entropy MXene Structure Optimization

This script performs Monte Carlo simulations with element substitutions for High Entropy Materials.
It integrates with VASP for energy calculations and ensures unique configurations are explored to find the minimum energy configuration (most stable configuration). This script can be adopted for any high entropy material systems.

Author: Noah Oyeniran
License: MIT
"""
from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Poscar
import os
import subprocess
import random
import numpy as np
import hashlib
import matplotlib.pyplot as plt

# Constants
MC_TEMPERATURE = 300  # Temperature in Kelvin
BOLTZMANN_CONSTANT = 8.617333262145e-5  # eV/K
MAX_ITERATIONS = 5000  # Maximum Monte Carlo iterations
POTCAR_PATH = "path/to/potpaw_PBE"
ELEMENT_POTCAR_MAP = {"Ti": "Ti_sv", "Mo": "Mo_sv", "Cr": "Cr_pv", "Nb": "Nb_sv", "C": "C", "N": "N", "F": "F", "O": "O"}

def reorder_species_concisely(structure):
    """Reorders species in a structure for consistent output."""
    composition = structure.composition
    sorted_elements = sorted(composition.as_dict().keys())
    reordered_sites = []
    for element in sorted_elements:
        element_sites = [site for site in structure if site.species_string == element]
        reordered_sites.extend(element_sites)
    return Structure.from_sites(reordered_sites)

def create_supercell_and_substitute(poscar_file, scaling_matrix, substitutions, x_elements, output_dir):
    """Creates a supercell, substitutes elements, and saves the modified structure."""
    structure = Structure.from_file(poscar_file)
    structure.make_supercell(scaling_matrix)

    x_elements = [Element(el) for el in x_elements]
    for element_to_replace, substituent_data in substitutions.items():
        replaceable_sites = [
            i for i, site in enumerate(structure)
            if site.species_string == element_to_replace and Element(site.species_string) not in x_elements
        ]
        
        total_sites = len(replaceable_sites)
        random.shuffle(replaceable_sites)
        start_index = 0

        for substituent_element, mole_fraction in substituent_data.items():
            num_to_substitute = int(mole_fraction * total_sites)
            sites_to_substitute = replaceable_sites[start_index:start_index + num_to_substitute]
            for site_index in sites_to_substitute:
                structure[site_index] = Element(substituent_element)
            start_index += num_to_substitute

    structure = reorder_species_concisely(structure)
    os.makedirs(output_dir, exist_ok=True)
    poscar_path = os.path.join(output_dir, "POSCAR")
    Poscar(structure).write_file(poscar_path)
    return poscar_path

def create_potcar(poscar_order, output_dir):
    """Creates a POTCAR file based on the POSCAR order."""
    unique_elements = list(dict.fromkeys(poscar_order))
    potcar_files = []
    for element in unique_elements:
        potcar_filename = ELEMENT_POTCAR_MAP.get(element)
        if potcar_filename:
            potcar_files.append(os.path.join(POTCAR_PATH, potcar_filename, "POTCAR"))
        else:
            raise ValueError(f"No POTCAR available for element {element}")

    output_potcar_path = os.path.join(output_dir, "POTCAR")
    with open(output_potcar_path, 'wb') as outfile:
        for potcar_path in potcar_files:
            if not os.path.exists(potcar_path):
                raise FileNotFoundError(f"Missing POTCAR file for {element}")
            with open(potcar_path, 'rb') as infile:
                outfile.write(infile.read())

def generate_structure_fingerprint(structure):
    """Generates a unique fingerprint for the structure."""
    site_data = []
    for site in structure:
        neighbors = structure.get_neighbors(site, r=3.0)
        coordination_numbers = len(neighbors)
        species_count = {nbr.species_string: 0 for nbr in neighbors}
        for neighbor in neighbors:
            species_count[neighbor.species_string] += 1
        site_data.append((site.species_string, coordination_numbers, species_count))
    return hashlib.md5(str(site_data).encode()).hexdigest()

def run_vasp(directory):
    """Runs VASP in the given directory and extracts energy from OUTCAR."""
    with open(os.path.join(directory, 'INCAR'), 'w') as incar:
        incar.write("SYSTEM = MXene Structure Optimization\nENCUT = 500\nISMEAR = 0; SIGMA = 0.05\nEDIFF = 1E-3\nIBRION = 2\nNSW = 0\nISIF = 2\nIVDW = 10\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n")

    with open(os.path.join(directory, 'KPOINTS'), 'w') as kpoints:
        kpoints.write("Automatic mesh\n0\nMonkhorst-Pack\n2 2 1\n0 0 0\n")

    vasp_cmd = "path/to/vasp_executable"
    result = subprocess.run(vasp_cmd, shell=True, cwd=directory)

    if result.returncode != 0:
        raise RuntimeError("VASP execution failed.")

    with open(os.path.join(directory, "OUTCAR"), "r") as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                return float(line.split()[-2])
    raise ValueError("Energy not found in OUTCAR.")

def enhanced_monte_carlo_simulation(poscar_file, scaling_matrix, substitutions, x_elements, output_dir):
    """Performs Monte Carlo simulation for structure optimization."""
    substituted_poscar_path = create_supercell_and_substitute(
        poscar_file, scaling_matrix, substitutions, x_elements, output_dir
    )

    structure = Structure.from_file(substituted_poscar_path)
    energies = []
    unique_structures = set()
    energy_trend_path = os.path.join(output_dir, "energy_trend.txt")
    discarded_dir = os.path.join(output_dir, "discarded")
    discarded_trend_path = os.path.join(discarded_dir, "discarded_energies.txt")

    os.makedirs(discarded_dir, exist_ok=True)

    with open(energy_trend_path, "w") as energy_file:
        energy_file.write("Iteration\tEnergy (eV)\n")

    with open(discarded_trend_path, "w") as discarded_file:
        discarded_file.write("Iteration\tEnergy (eV)\n")

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}")

        # Regenerate structure if current one is a duplicate
        fingerprint = generate_structure_fingerprint(structure)
        if fingerprint in unique_structures:
            print("Duplicate structure detected, regenerating...")
            substituted_poscar_path = create_supercell_and_substitute(
                poscar_file, scaling_matrix, substitutions, x_elements, output_dir
            )
            structure = Structure.from_file(substituted_poscar_path)
            continue

        unique_structures.add(fingerprint)

        poscar_path = os.path.join(output_dir, 'POSCAR')
        Poscar(structure).write_file(poscar_path)
        create_potcar([site.species_string for site in structure.sites], output_dir)

        try:
            energy = run_vasp(output_dir)
        except Exception as e:
            print(f"VASP run failed: {e}")
            continue

        if iteration > 0 and energy > energies[-1]:
            print("Higher energy configuration discarded.")
            discarded_contcar_path = os.path.join(discarded_dir, f"CONTCAR_{iteration + 1}")
            structure.to(fmt="poscar", filename=discarded_contcar_path)

            with open(discarded_trend_path, "a") as discarded_file:
                discarded_file.write(f"{iteration + 1}\t{energy}\n")

            continue

        energies.append(energy)

        with open(energy_trend_path, "a") as energy_file:
            energy_file.write(f"{iteration + 1}\t{energy}\n")

        contcar_path = os.path.join(output_dir, f"CONTCAR_{iteration + 1}")
        structure.to(fmt="poscar", filename=contcar_path)

        if iteration > 0:
            delta_energy = energy - energies[-2]
            acceptance_probability = min(1.0, np.exp(-delta_energy / (BOLTZMANN_CONSTANT * MC_TEMPERATURE)))
            if random.uniform(0, 1) > acceptance_probability:
                print("Rejected move, reverting to previous structure.")
                structure = Structure.from_file(contcar_path)
                continue

    plt.plot(range(1, len(energies) + 1), energies, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (eV)")
    plt.title("Energy vs Iteration")
    plt.savefig(os.path.join(output_dir, "energy_trend.png"))

# Run Simulation
enhanced_monte_carlo_simulation(
    poscar_file="POSCAR",
    scaling_matrix=[3, 3, 1], #This is for the supercell size
    #substitutions={"Mo": {"Nb": 0.5, "Cr": 0.3, "Ti": 0.2}},
    substitutions={"Mo": {"Cr": 0.33, "Ti": 0.33}},  # The first element before second curl bracket is the element in the initial POSCAR while the other elements are element to substitute with their vf.
    x_elements=["C", "N", "F", "O"],   #Position of these elements are fixed. It includes both the X elements and surface terminated elements.
    output_dir="mc_output"
)
