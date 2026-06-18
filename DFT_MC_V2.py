# -*- coding: utf-8 -*-
#Version:V2
#Author: Noah Oyeniran
#Hu research group
#Aero. Engr. and Mech. The University of Alabama

from pymatgen.core import Structure, Element
from pymatgen.io.vasp import Poscar
import os
import subprocess
import random
import numpy as np
import hashlib
import matplotlib.pyplot as plt

# =========================================================
# CONSTANTS
# =========================================================

MC_TEMPERATURE = 300  # Kelvin

KB = 8.617333262145e-5  # eV/K

MAX_ITERATIONS = 10000

RELAXATION_INTERVAL = 30

RELAX_NSW = 3

# =========================================================
# POTCAR SETTINGS
# =========================================================

POTCAR_PATH = "/path/to/POTCAR_file"

ELEMENT_POTCAR_MAP = {
    "Ti": "Ti_sv",
    "Mo": "Mo_sv",
    "Cr": "Cr_pv",
    "V": "V_sv",
    "Nb": "Nb_sv",
    "Mn": "Mn_pv",
    "Hf": "Hf_pv",
    "Ta": "Ta_pv",
    "Re": "Re",
    "C": "C",
    "N": "N",
    "O": "O",
    "F": "F"
}

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def reorder_species_concisely(structure):

    composition = structure.composition

    sorted_elements = sorted(
        composition.as_dict().keys()
    )

    reordered_sites = []

    for element in sorted_elements:

        element_sites = [
            site for site in structure
            if site.species_string == element
        ]

        reordered_sites.extend(element_sites)

    return Structure.from_sites(reordered_sites)

# =========================================================
# CREATE INITIAL STRUCTURE
# =========================================================

def create_supercell_and_substitute(
        poscar_file,
        scaling_matrix,
        base_element,
        substituents,
        mole_fractions,
        x_elements,
        output_dir):

    structure = Structure.from_file(poscar_file)

    structure.make_supercell(scaling_matrix)

    x_elements = [Element(el) for el in x_elements]

    replaceable_sites = [

        i for i, site in enumerate(structure)

        if site.species_string == base_element
        and Element(site.species_string) not in x_elements
    ]

    total_sites = len(replaceable_sites)

    assigned_sites = []

    # =====================================================
    # MULTICOMPONENT SUBSTITUTION
    # =====================================================

    for element, frac in zip(substituents, mole_fractions):

        #n_sub = int(frac * total_sites)
        n_sub = round(frac * total_sites)

        available_sites = [
            i for i in replaceable_sites
            if i not in assigned_sites
        ]

        chosen_sites = random.sample(
            available_sites,
            n_sub
        )

        for idx in chosen_sites:
            structure[idx] = Element(element)

        assigned_sites.extend(chosen_sites)

    structure = reorder_species_concisely(structure)

    os.makedirs(output_dir, exist_ok=True)

    Poscar(structure).write_file(
        os.path.join(output_dir, "POSCAR")
    )

    return structure

# =========================================================
# CREATE POTCAR
# =========================================================

def create_potcar(poscar_order, output_dir):

    unique_elements = list(
        dict.fromkeys(poscar_order)
    )

    potcar_files = []

    for element in unique_elements:

        potcar_filename = ELEMENT_POTCAR_MAP.get(element)

        if potcar_filename:

            potcar_files.append(
                os.path.join(
                    POTCAR_PATH,
                    potcar_filename,
                    "POTCAR"
                )
            )

        else:
            raise ValueError(
                f"No POTCAR available for element {element}"
            )

    output_potcar_path = os.path.join(
        output_dir,
        "POTCAR"
    )

    with open(output_potcar_path, 'wb') as outfile:

        for potcar_path in potcar_files:

            if not os.path.exists(potcar_path):

                raise FileNotFoundError(
                    f"Missing POTCAR: {potcar_path}"
                )

            with open(potcar_path, 'rb') as infile:
                outfile.write(infile.read())

# =========================================================
# FINGERPRINT
# =========================================================

def generate_structure_fingerprint(structure):

    site_data = []

    for site in structure:

        neighbors = structure.get_neighbors(site, r=3.0)

        coord = len(neighbors)

        species_count = {}

        for nbr in neighbors:

            el = nbr.species_string

            if el not in species_count:
                species_count[el] = 0

            species_count[el] += 1

        site_data.append(
            (
                site.species_string,
                tuple(np.round(site.frac_coords, 3)),
                coord,
                tuple(sorted(species_count.items()))
            )
        )

    return hashlib.md5(
        str(site_data).encode()
    ).hexdigest()

# =========================================================
# RANDOM SWAP MOVE
# =========================================================

def swap_random_atoms(structure, x_elements):

    new_structure = structure.copy()

    metal_indices = [

        i for i, site in enumerate(new_structure)

        if site.species_string not in x_elements
    ]

    for _ in range(20):

        i, j = random.sample(metal_indices, 2)

        if (
            new_structure[i].species_string
            !=
            new_structure[j].species_string
        ):

            species_i = new_structure[i].species_string
            species_j = new_structure[j].species_string

            new_structure[i] = Element(species_j)
            new_structure[j] = Element(species_i)

            return new_structure

    return new_structure

def structure_is_valid(structure):

    distance_matrix = structure.distance_matrix

    n = len(distance_matrix)

    for i in range(n):

        for j in range(i + 1, n):

            specie_i = structure[i].species_string
            specie_j = structure[j].species_string

            d = distance_matrix[i][j]

            # Metal-metal
            if specie_i not in ["C", "N", "O", "F"] \
               and specie_j not in ["C", "N", "O", "F"]:

                if d < 2.0:
                    return False

            # Metal-X
            else:

                if d < 1.5:
                    return False

    return True

# =========================================================
# RUN VASP
# =========================================================

def run_vasp(directory, relax=False):

    # =====================================================
    # CLEAN PREVIOUS FILES
    # =====================================================

    cleanup_files = [

        "WAVECAR",
        "CHGCAR",
        "vasprun.xml",
        "EIGENVAL",
        "DOSCAR",
        "IBZKPT",
        "XDATCAR",
        "REPORT"

    ]

    for fname in cleanup_files:

        fpath = os.path.join(directory, fname)

        if os.path.exists(fpath):

            try:
                os.remove(fpath)
            except:
                pass

    # =====================================================
    # INCAR
    # =====================================================

    incar_path = os.path.join(directory, "INCAR")

    with open(incar_path, "w") as incar:

        incar.write("SYSTEM = HE-MXene Monte Carlo\n")

        incar.write("ENCUT = 500\n")

        incar.write("EDIFF = 1E-4\n") #incar.write("NELM = 200\n")

        incar.write("ALGO = Fast\n") #incar.write("ALGO = Normal\n")

        incar.write("ISMEAR = 0\n")

        incar.write("SIGMA = 0.05\n") #incar.write("ISYM = 0\n"); can turn symmetric off 

        incar.write("IBRION = 2\n") #very fast when IBRION is set to -1 and NSW=0 for self consistent calculation

        incar.write("ISIF = 2\n")

        incar.write("POTIM = 0.10\n")

        incar.write("LASPH = .TRUE.\n")

        incar.write("ADDGRID = .TRUE.\n") #incar.write("LREAL = Auto\n"); may include

        incar.write("IVDW = 10\n")

        incar.write("LWAVE = .FALSE.\n")

        incar.write("LCHARG = .FALSE.\n")

        incar.write("PREC = Accurate\n")

        incar.write("KPAR = 2\n")

        if relax:

            incar.write(f"NSW = {RELAX_NSW}\n")
            #incar.write("IBRION = 2\n"); this is just a place holder you can use to ensure good relaxation and speed

        else:

            incar.write("NSW = 0\n")
            #incar.write("IBRION = -1\n"); this is just a place holder you can use to ensure speed

    # =====================================================
    # KPOINTS
    # =====================================================

    with open(os.path.join(directory, "KPOINTS"), "w") as kpoints:

        kpoints.write("Automatic mesh\n")

        kpoints.write("0\n")

        kpoints.write("Monkhorst-Pack\n")

        kpoints.write("2 2 1\n")

        kpoints.write("0 0 0\n")

    # =====================================================
    # RUN VASP
    # =====================================================

    vasp_cmd = (

        "time srun -n8 -c32 --cpu-bind=cores -G8 "
        "/path/to/vasp/installation/packages/for_users/"
        "vasp.6.4.1-2d-gpu/bin/vasp_std > stdout 2>&1"
    )

    result = subprocess.run(

        vasp_cmd,
        shell=True,
        cwd=directory # You can include this as well, timeout=7200
    )

    # =====================================================
    # CHECK FAILURE
    # =====================================================

    if result.returncode != 0:

        raise RuntimeError(
            f"VASP execution failed with code {result.returncode}"
        )

    outcar_path = os.path.join(directory, "OUTCAR")

    if not os.path.exists(outcar_path):

        raise RuntimeError("OUTCAR missing.")

    # =====================================================
    # CHECK FOR NUMERICAL CRASH
    # =====================================================

    with open(outcar_path, "r") as f:

        outcar_text = f.read()

    crash_keywords = [

        "ZBRENT",
        "LAPACK",
        "BRMIX",
        "VERY BAD NEWS",
        "WARNING: Sub-Space-Matrix",
        "EDDDAV",
        "DAV:",
        "ERROR",
        "NaN"

    ]

    for key in crash_keywords:

        if key in outcar_text:

            print(f"[WARNING] Possible instability detected: {key}")

    # =====================================================
    # EXTRACT ENERGY
    # =====================================================

    energy = None

    for line in outcar_text.splitlines():

        if "free  energy   TOTEN" in line:

            try:
                energy = float(line.split()[-2])
            except:
                pass

    if energy is None:

        raise RuntimeError(
            "Energy not found in OUTCAR."
        )

    if not np.isfinite(energy):

        raise RuntimeError(
            "Non-finite energy encountered."
        )

    return energy

# =========================================================
# MAIN MONTE CARLO
# =========================================================

def enhanced_monte_carlo_simulation(
        poscar_file,
        scaling_matrix,
        base_element,
        substituents,
        mole_fractions,
        x_elements,
        output_dir):

    os.makedirs(output_dir, exist_ok=True)

    discarded_dir = os.path.join(
        output_dir,
        "discarded"
    )

    os.makedirs(discarded_dir, exist_ok=True)

    # =====================================================
    # INITIAL STRUCTURE
    # =====================================================

    structure = create_supercell_and_substitute(
        poscar_file,
        scaling_matrix,
        base_element,
        substituents,
        mole_fractions,
        x_elements,
        output_dir
    )

    # =====================================================
    # INITIAL RELAXATION
    # =====================================================

    Poscar(
        reorder_species_concisely(structure)
    ).write_file(
        os.path.join(output_dir, "POSCAR")
    )

    create_potcar(
        [site.species_string for site in structure],
        output_dir
    )

    current_energy = run_vasp(
        output_dir,
        relax=True
    )

    contcar = os.path.join(output_dir, "CONTCAR")

    if os.path.exists(contcar):

        structure = Structure.from_file(contcar)

    best_energy = current_energy

    best_structure = structure.copy()

    accepted_structure = structure.copy()

    accepted_energy = current_energy

    energies = [current_energy]

    unique_structures = set()

    fingerprint_file = os.path.join(
        output_dir,
        "all_fingerprints.txt"
    )

    if os.path.exists(fingerprint_file):

        with open(fingerprint_file, "r") as fp:

            for line in fp:

                unique_structures.add(
                    line.strip()
                )

    initial_fp = generate_structure_fingerprint(
        structure
    )

    unique_structures.add(initial_fp)

    with open(fingerprint_file, "a") as fp:

        fp.write(f"{initial_fp}\n")

    # =====================================================
    # LOG FILES
    # =====================================================

    energy_trend_path = os.path.join(
        output_dir,
        "energy_trend.txt"
    )

    discarded_trend_path = os.path.join(
        discarded_dir,
        "discarded_energies.txt"
    )

    with open(energy_trend_path, "w") as f:
        f.write("Iteration\tEnergy(eV)\n")

    with open(discarded_trend_path, "w") as f:
        f.write("Iteration\tEnergy(eV)\n")

    # =====================================================
    # MONTE CARLO LOOP
    # =====================================================

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\nIteration {iteration}")

        # =================================================
        # GENERATE TRIAL MOVE
        # =================================================

        trial_structure = swap_random_atoms(
            accepted_structure,
            x_elements
        )

        fingerprint = generate_structure_fingerprint(
            trial_structure
        )

        if not structure_is_valid(trial_structure):

            print("Invalid structure skipped.")

            continue

        if fingerprint in unique_structures:

            print("Duplicate structure skipped.")

            continue

        # =================================================
        # WRITE POSCAR
        # =================================================

        Poscar(
            reorder_species_concisely(trial_structure)
        ).write_file(
            os.path.join(output_dir, "POSCAR")
        )

        create_potcar(
            [site.species_string for site in trial_structure],
            output_dir
        )

        # =================================================
        # ENERGY EVALUATION (SAFE BLOCK)
        # =================================================

        relax_step = (
            iteration % RELAXATION_INTERVAL == 0
        )

        try:

            trial_energy = run_vasp(
                output_dir,
                relax=relax_step
            )

            # =============================================
            # LOAD RELAXED STRUCTURE
            # =============================================

            if relax_step:

                contcar = os.path.join(
                    output_dir,
                    "CONTCAR"
                )

                if os.path.exists(contcar):

                    relaxed_structure = Structure.from_file(
                        contcar
                    )

                    if structure_is_valid(
                        relaxed_structure
                    ):

                        trial_structure = (
                            relaxed_structure
                        )

                    else:

                        print(
                            "Relaxed structure became invalid."
                        )

                        continue

            # =============================================
            # ENERGY SAFETY CHECK
            # =============================================

            if (
                trial_energy is None
                or
                not np.isfinite(trial_energy)
            ):

                print(
                    "[WARNING] Invalid energy encountered."
                )

                continue

        except subprocess.TimeoutExpired:

            print(
                f"[WARNING] Iteration {iteration} "
                "timed out."
            )

            continue

        except Exception as e:

            print(
                f"[WARNING] VASP crashed at "
                f"iteration {iteration}: {e}"
            )

            continue

        # =================================================
        # METROPOLIS CRITERION
        # =================================================

        delta_energy = (
            trial_energy - accepted_energy
        )

        exponent = (
            -delta_energy /
            (KB * MC_TEMPERATURE)
        )

        # Numerical stability
        exponent = np.clip(exponent, -700, 700)

        if delta_energy <= 0:

            acceptance_probability = 1.0

        else:

            acceptance_probability = np.exp(exponent)

        rand_num = random.random()

        print(f"dE = {delta_energy:.6f} eV")

        print(f"P_accept = {acceptance_probability:.6e}")

        # =================================================
        # ACCEPT MOVE
        # =================================================

        if rand_num < acceptance_probability:

            print("Move accepted.")

            accepted_structure = trial_structure.copy()

            accepted_energy = trial_energy
            unique_structures.add(fingerprint)
            with open(
                os.path.join(output_dir, "all_fingerprints.txt"),
                "a"
            ) as fp:

                fp.write(f"{fingerprint}\n")

            energies.append(trial_energy)

            with open(energy_trend_path, "a") as f:

                f.write(
                    f"{iteration}\t{trial_energy}\n"
                )

            Poscar(
                reorder_species_concisely(
                    accepted_structure
                )
            ).write_file(
                os.path.join(
                    output_dir,
                    f"CONTCAR_{iteration}"
                )
            )

            # =============================================
            # BEST STRUCTURE
            # =============================================

            if trial_energy < best_energy:

                best_energy = trial_energy

                best_structure = trial_structure.copy()

                Poscar(
                    reorder_species_concisely(
                        best_structure
                    )
                ).write_file(
                    os.path.join(
                        output_dir,
                        "BEST_CONTCAR"
                    )
                )

                with open(
                    os.path.join(
                        output_dir,
                        "best_energy.txt"
                    ),
                    "w"
                ) as f:

                    f.write(
                        f"{best_energy}\n"
                    )

        # =================================================
        # REJECT MOVE
        # =================================================

        else:

            print("Move rejected.")

            rejected_path = os.path.join(
                discarded_dir,
                f"REJECTED_{iteration}"
            )

            Poscar(
                reorder_species_concisely(
                    trial_structure
                )
            ).write_file(rejected_path)

            with open(discarded_trend_path, "a") as f:

                f.write(
                    f"{iteration}\t{trial_energy}\n"
                )
            with open(
                os.path.join(
                    discarded_dir,
                    "rejected_fingerprints.txt"
                ),
                "a"
            ) as fp:

                fp.write(f"{fingerprint}\n")

    # =====================================================
    # FINAL PLOT
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(len(energies)),
        energies,
        marker='o'
    )

    plt.xlabel("Accepted MC Step")

    plt.ylabel("Energy (eV)")

    plt.title("Monte Carlo Energy Evolution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "energy_trend.png"
        )
    )

    return energies

# =========================================================
# RUN SIMULATION
# =========================================================

energies = enhanced_monte_carlo_simulation(

    poscar_file="POSCAR",

    scaling_matrix=[4, 4, 1],

    base_element="Ti",

    substituents=[
        "Mo",
        "V",
        "Nb"
    ],

    mole_fractions=[
        0.50,
        0.00,
        0.00
    ],

    x_elements=[
        "C",
        "N",
        "O",
        "F"
    ],

    output_dir="mc_output"

)