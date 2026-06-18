# -*- coding: utf-8 -*-
#Version:V2
#Author: Noah Oyeniran
#Hu research group
#Aero. Engr. and Mech. The University of Alabama

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import os
import numpy as np
import hashlib
import subprocess
import random
import matplotlib.pyplot as plt

# ============================================================
# Monte Carlo + VASP Continuation Code
# Robust Metropolis Criterion + Numerical Stability
# ============================================================

# ===== Constants & knobs =====

MC_TEMPERATURE = 300  # Kelvin

BOLTZMANN_CONSTANT = 8.617333262145e-5  # eV/K

MAX_CONTINUATION_ITERATIONS = 10000

RELAXATION_INTERVAL = 20

RELAX_NSW = 10

HEAL_ON_START = False

ENERGY_TOL = 1e-8

POTCAR_PATH = "/path/to/where/POTCAR/is/located"

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
    "F": "F",
    "Ni": "Ni"
}

# Only transition-metal sublattice swaps
SWAP_ELEMENTS = [
    "Ti", "Mo", "Cr", "V",
    "Nb", "Mn", "Hf", "Ta",
    "Re", "Ni"
]

# ============================================================
# Utility Functions
# ============================================================

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
   
    return hashlib.md5(str(site_data).encode()).hexdigest()


def _write_incar(directory, relax=False):

    with open(os.path.join(directory, 'INCAR'), 'w') as incar:

        incar.write("SYSTEM = MXene_MC\n")

        incar.write("ENCUT = 400\n")

        incar.write("ISMEAR = 0\n")

        incar.write("SIGMA = 0.05\n")

        incar.write("EDIFF = 1E-4\n")

        incar.write("IBRION = 2\n") # This can be changed to -1 for speed. Same as in the main script

        incar.write("ISIF = 2\n")

        incar.write("PREC = Accurate\n")

        incar.write("LWAVE = .FALSE.\n")

        incar.write("LCHARG = .FALSE.\n")

        incar.write("KPAR = 2\n")

        incar.write("IVDW = 10\n")

        incar.write("NELM = 100\n")

        incar.write("ALGO = Fast\n")

        incar.write("LASPH = .TRUE.\n")

        incar.write("ADDGRID = .TRUE.\n")

        if relax:
            incar.write(f"NSW = {RELAX_NSW}\n")
        else:
            incar.write("NSW = 0\n")


def _write_kpoints(directory):

    with open(os.path.join(directory, 'KPOINTS'), 'w') as kpoints:

        kpoints.write("Automatic mesh\n")

        kpoints.write("0\n")

        kpoints.write("Monkhorst-Pack\n")

        kpoints.write("2 2 1\n")

        kpoints.write("0 0 0\n")


def _read_energy_from_outcar(directory):

    outcar_path = os.path.join(directory, "OUTCAR")

    last_energy = None

    with open(outcar_path) as f:

        for line in f:

            if "free  energy   TOTEN" in line:

                try:
                    last_energy = float(line.split()[-2])

                except:
                    pass

    if last_energy is None:
        raise ValueError("Energy not found in OUTCAR.")

    if not np.isfinite(last_energy):
        raise ValueError("Invalid energy detected.")

    return last_energy


def _read_contcar_or_poscar(directory):

    contcar = os.path.join(directory, "CONTCAR")

    poscar = os.path.join(directory, "POSCAR")

    if os.path.exists(contcar) and os.path.getsize(contcar) > 0:

        return Structure.from_file(contcar)

    return Structure.from_file(poscar)


def run_vasp(directory, relax=False):

    _write_incar(directory, relax=relax)

    _write_kpoints(directory)

    vasp_cmd = (
        "time srun -n8 -c32 --cpu-bind=cores -G8 "
        "/path/to/where/your/vasp/located/"
        "specific_version/bin/vasp_std > stdout 2>&1"
    )

    result = subprocess.run(
        vasp_cmd,
        shell=True,
        cwd=directory
    )

    if result.returncode != 0:
        raise RuntimeError("VASP execution failed")

    energy = _read_energy_from_outcar(directory)

    structure_out = _read_contcar_or_poscar(directory)

    return energy, structure_out


def create_potcar(poscar_order, output_dir):

    unique_elements = list(dict.fromkeys(poscar_order))

    potcar_files = []

    for el in unique_elements:

        filename = ELEMENT_POTCAR_MAP.get(el)

        if filename:

            potcar_files.append(
                os.path.join(
                    POTCAR_PATH,
                    filename,
                    "POTCAR"
                )
            )

        else:
            raise ValueError(f"No POTCAR for {el}")

    with open(os.path.join(output_dir, "POTCAR"), 'wb') as outfile:

        for path in potcar_files:

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"POTCAR not found at {path}"
                )

            with open(path, 'rb') as infile:

                outfile.write(infile.read())


def load_energy_trend(path):

    iterations = []

    energies = []

    if not os.path.exists(path):

        print(f"{path} not found.")

        fallback_path = os.path.join(
            "mc_output",
            "energy_trend.txt"
        )

        if not os.path.exists(fallback_path):

            raise FileNotFoundError(
                "No energy trend file found."
            )

        path = fallback_path

    with open(path, 'r') as f:

        header = f.readline()

        for line in f:

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            iterations.append(int(parts[0]))

            energies.append(float(parts[1]))

    return iterations, energies


def _ordered_by_first_appearance(structure):

    element_order = []

    seen = set()

    for site in structure.sites:

        el = site.species_string

        if el not in seen:

            seen.add(el)

            element_order.append(el)

    sites_by_el = {el: [] for el in element_order}

    for site in structure.sites:

        sites_by_el[site.species_string].append(site)

    ordered_sites = []

    for el in element_order:

        ordered_sites.extend(sites_by_el[el])

    ordered_structure = Structure(
        structure.lattice,
        [site.species for site in ordered_sites],
        [site.frac_coords for site in ordered_sites]
    )

    return ordered_structure, element_order

# ============================================================
# Main MC Continuation
# ============================================================

def continue_simulation(output_dir):

    cont_dir = os.path.join(output_dir, "continued")

    os.makedirs(cont_dir, exist_ok=True)

    discarded_dir = os.path.join(cont_dir, "discarded")

    os.makedirs(discarded_dir, exist_ok=True)

    energy_trend_path = os.path.join(
        cont_dir,
        "energy_trend_continued.txt"
    )

    discarded_path = os.path.join(
        discarded_dir,
        "discarded_energies.txt"
    )

    iterations, energies = load_energy_trend(
        energy_trend_path
    )

    current_iter = iterations[-1]

    prev_energy = energies[-1]

    last_contcar = os.path.join(
        output_dir,
        f"CONTCAR_{current_iter}"
    )

    structure = Structure.from_file(last_contcar)

    unique_structures = set()

    fingerprint_file = os.path.join(
        output_dir,
        "all_fingerprints.txt"
    )

    if os.path.exists(fingerprint_file):

        with open(fingerprint_file, "r") as fp:

            for line in fp:

                try:

                    unique_structures.add(
                        line.strip()
                    )

                except:

                    pass

    if HEAL_ON_START:

        print("Performing initial healing relaxation...")

        ordered_structure, el_order = \
            _ordered_by_first_appearance(structure)

        Poscar(
            ordered_structure,
            sort_structure=False
        ).write_file(
            os.path.join(cont_dir, "POSCAR")
        )

        create_potcar(el_order, cont_dir)

        try:

            healed_energy, healed_structure = \
                run_vasp(cont_dir, relax=True)

            if healed_energy <= prev_energy:

                print("Healing accepted.")

                prev_energy = healed_energy

                structure = healed_structure

            else:

                print("Healing rejected.")

        except Exception as e:

            print(f"Healing failed: {e}")

    # ========================================================
    # Monte Carlo Loop
    # ========================================================

    for step in range(1, MAX_CONTINUATION_ITERATIONS + 1):

        new_iter = current_iter + step

        print("=" * 60)

        print(f"MC Iteration {new_iter}")

        mod_structure = structure.copy()

        modifiable_indices = [

            i for i, site in enumerate(mod_structure)

            if site.species_string in SWAP_ELEMENTS
        ]

        if len(modifiable_indices) < 2:

            print("Not enough swap candidates.")

            continue

        i1, i2 = random.sample(modifiable_indices, 2)

        species1 = mod_structure[i1].species

        species2 = mod_structure[i2].species

        mod_structure[i1].species = species2

        mod_structure[i2].species = species1

        fingerprint = generate_structure_fingerprint(
            mod_structure
        )

        if fingerprint in unique_structures:

            print("Duplicate structure detected.")

            continue

        ordered_structure, element_order = \
            _ordered_by_first_appearance(mod_structure)

        Poscar(
            ordered_structure,
            sort_structure=False
        ).write_file(
            os.path.join(cont_dir, "POSCAR")
        )

        create_potcar(element_order, cont_dir)

        do_relax = (
            new_iter % RELAXATION_INTERVAL == 0
        )

        try:

            new_energy, resulting_structure = \
                run_vasp(
                    cont_dir,
                    relax=do_relax
                )

        except Exception as e:

            print(f"VASP failed: {e}")

            continue

        # ====================================================
        # Robust Metropolis Criterion
        # ====================================================

        delta_E = new_energy - prev_energy

        print(f"Previous Energy = {prev_energy:.8f} eV")

        print(f"New Energy      = {new_energy:.8f} eV")

        print(f"Delta E         = {delta_E:.8f} eV")

        accept_move = False

        if not np.isfinite(new_energy):

            print("Invalid energy encountered.")

            accept_move = False

        elif delta_E <= ENERGY_TOL:

            print("Downhill move accepted.")

            accept_move = True

        else:

            try:

                beta = 1.0 / (
                    BOLTZMANN_CONSTANT *
                    MC_TEMPERATURE
                )

                exponent = -delta_E * beta

                exponent = np.clip(
                    exponent,
                    -700,
                    700
                )

                acceptance_probability = np.exp(
                    exponent
                )

                random_number = random.random()

                print(
                    f"Acceptance Probability = "
                    f"{acceptance_probability:.6e}"
                )

                print(
                    f"Random Number = "
                    f"{random_number:.6f}"
                )

                if random_number < acceptance_probability:

                    print("Uphill move accepted.")

                    accept_move = True

                else:

                    print("Uphill move rejected.")

                    accept_move = False

            except OverflowError:

                print("Overflow encountered.")

                accept_move = False

        # ====================================================
        # Apply Acceptance/Rejection
        # ====================================================

        if accept_move:

            energies.append(new_energy)

            prev_energy = new_energy

            structure = resulting_structure

            unique_structures.add(fingerprint)

            with open(
                os.path.join(
                    output_dir,
                    "all_fingerprints.txt"
                ),
                "a"
            ) as fpfile:

                fpfile.write(f"{fingerprint}\n")

            with open(
                energy_trend_path,
                'a'
            ) as ef:

                ef.write(
                    f"{new_iter}\t{new_energy}\n"
                )

            Poscar(
                structure,
                sort_structure=False
            ).write_file(
                os.path.join(
                    cont_dir,
                    f"CONTCAR_{new_iter}"
                )
            )

            print("Move accepted.")

        else:

            Poscar(
                ordered_structure,
                sort_structure=False
            ).write_file(
                os.path.join(
                    discarded_dir,
                    f"CONTCAR_{new_iter}"
                )
            )

            with open(
                discarded_path,
                'a'
            ) as df:

                df.write(
                    f"{new_iter}\t{new_energy}\n"
                )

            print("Move rejected.")

    # ========================================================
    # Energy Plot
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, len(energies) + 1),
        energies,
        marker='o'
    )

    plt.xlabel("Accepted MC Steps")

    plt.ylabel("Energy (eV)")

    plt.title("Monte Carlo Energy Evolution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            cont_dir,
            "energy_trend_continued.png"
        ),
        dpi=300
    )

    print("Simulation completed.")

# ============================================================
# Run Simulation
# ============================================================

continue_simulation(
    output_dir="mc_output"
)