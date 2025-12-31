import os

import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

import openmmnqe as nqe


def test_run_openmm_relaxation():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_relaxation(modeller, forcefield, platform_name='CUDA')
    os.remove('minimized.pdb')


def test_run_openmm_heating():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield)
    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')


def test_run_openmm_heating_deuterate():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_heating(modeller, forcefield, deuterate=True)
    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')


def test_run_openmm_npt():
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')
    nqe.run_openmm_npt(modeller, forcefield)
    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


def test_eq_workflow():
    print(flush=True)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    pdb = app.PDBFile("tests/data/pdb/input.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='dodecahedron')

    nqe.run_openmm_relaxation(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    os.remove('minimized.pdb')

    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')

    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


def test_eq_workflow_mixed():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 2.0
    box_shape = 'dodecahedron'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)

    chains = list(modeller.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    nqe.run_openmm_relaxation(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    pdb = app.PDBFile("equilibrated.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield, potential=potential, ml_idx=ml_atoms)

    os.remove('minimized.pdb')

    os.remove('equilibrate.chk')
    os.remove('equilibrate.log')
    os.remove('equilibrate.pdb')
    os.remove('equilibrate_steps.pdb')

    os.remove('npt_equilibrated.chk')
    os.remove('npt_equilibrated.log')
    os.remove('npt_equilibrated.pdb')
    os.remove('npt_equilibrated_steps.pdb')


from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlumedFES:
    data: np.ndarray  # numeric columns (after dropping der_* if possible)
    fields: List[str]  # matching names (after dropping der_*), may be []


def load_plumed_fes_drop_der(path: str) -> PlumedFES:
    fields_raw: List[str] = []
    numeric_lines: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#!"):
                parts = s.split()
                if len(parts) >= 3 and parts[1] == "FIELDS":
                    fields_raw = parts[2:]
                continue

            if s.startswith("#") or s.startswith("@"):
                continue

            numeric_lines.append(s)

    if not numeric_lines:
        raise ValueError(f"No numeric data found in {path}")

    data = np.loadtxt(numeric_lines)
    if data.ndim == 1:
        data = data[None, :]

    # Drop der_* only if fields align with data columns
    if fields_raw and len(fields_raw) == data.shape[1]:
        keep_idx = [i for i, name in enumerate(fields_raw) if not name.startswith("der_")]
        data = data[:, keep_idx]
        fields = [fields_raw[i] for i in keep_idx]
        return PlumedFES(data=data, fields=fields)

    return PlumedFES(data=data, fields=[])


def plot_plumed_fes(
        path: str,
        ax: Optional[plt.Axes] = None,
        shift_min_to_zero: bool = True,
        levels: int = 30,
) -> Tuple[plt.Figure, plt.Axes]:
    fes = load_plumed_fes_drop_der(path)
    data, fields = fes.data, fes.fields

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ncol = data.shape[1]
    if ncol < 2:
        raise ValueError(f"Need at least 2 columns to plot, got {ncol}")

    # Labels: use FIELDS if available, else defaults
    def lab(i: int, default: str) -> str:
        return fields[i] if fields and i < len(fields) else default

    if ncol == 2:
        x = data[:, 0]
        z = data[:, 1].copy()
        if shift_min_to_zero and np.isfinite(z).any():
            z -= np.nanmin(z)
        order = np.argsort(x)
        ax.plot(x[order], z[order])
        ax.set_xlabel(lab(0, "CV"))
        ax.set_ylabel(lab(1, "FES"))

    else:
        # Use first 3 columns for 2D plot even if more columns exist
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2].copy()

        if shift_min_to_zero and np.isfinite(z).any():
            z -= np.nanmin(z)

        xu = np.unique(x)
        yu = np.unique(y)

        if xu.size * yu.size == z.size:
            # regular grid: sort then reshape
            idx = np.lexsort((x, y))
            xs, ys, zs = x[idx], y[idx], z[idx]
            X = xs.reshape(yu.size, xu.size)
            Y = ys.reshape(yu.size, xu.size)
            Z = zs.reshape(yu.size, xu.size)
            m = ax.contourf(X, Y, Z, levels=levels)
        else:
            # irregular grid
            m = ax.tricontourf(x, y, z, levels=levels)

        ax.set_xlabel(lab(0, "CV1"))
        ax.set_ylabel(lab(1, "CV2"))
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label(lab(2, "FES"))

    return fig, ax


def test_eq_workflow_plumed_dihedral():
    print(flush=True)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    # Solvate
    padding = 2.0
    box_shape = 'cube'
    modeller.addSolvent(forcefield,
                        padding=padding * unit.nanometer,
                        boxShape=box_shape)
    idx = nqe.atom_indices_from_vmd_picks(modeller, ['ALA1:C', 'ALA2:N', 'ALA2:CA', 'ALA2:C'])
    print(f"Dihedral indices: {idx}", flush=True)
    idx_str = ",".join([str(i) for i in idx])

    plumed_input = f"""
# --- Define the torsion ---
phi: TORSION ATOMS={idx_str}

# --- Well-tempered metadynamics on the dihedral ---
# Units: default length=nm, energy=kJ/mol, time=ps (depends on your MD engine interface)
metad: METAD ARG=phi PACE=500 HEIGHT=1.2 SIGMA=0.35 BIASFACTOR=12 TEMP=300 GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 FILE=HILLS

# --- Monitor / output ---
PRINT STRIDE=200 ARG=phi,metad.bias FILE=COLVAR
"""
    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed_input)
    # Minimise the system first
    nqe.run_openmm_relaxation_simple(modeller, forcefield, platform_name='CUDA')
    #
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_prod(modeller, forcefield, plumed_script_path=plumed_script_path, platform_name='CUDA', steps=100_000)
    # Run PLUMED sum_hills to get FES
    os.system(f'plumed sum_hills --hills HILLS --outfile fes.dat --min -pi --max pi --bin 300 --kt 2.494')
    # Plot FES
    fig, ax = plot_plumed_fes("fes.dat")
    plt.show()

    os.remove('minimized.chk')
    os.remove('minimized.log')
    os.remove('minimized.pdb')
    os.remove('minimized_steps.pdb')

    # os.remove('equilibrate.chk')
    # os.remove('equilibrate.log')
    # os.remove('equilibrate.pdb')
    # os.remove('equilibrate_steps.pdb')
    #
    # os.remove('npt_equilibrated.chk')
    # os.remove('npt_equilibrated.log')
    # os.remove('npt_equilibrated.pdb')
    # os.remove('npt_equilibrated_steps.pdb')

    os.remove('prod.chk')
    os.remove('prod.log')
    os.remove('prod.pdb')
    os.remove('prod_steps.pdb')

    os.remove('COLVAR')
    os.remove('HILLS')
    os.remove('fes.dat')
    os.remove('plumed.dat')



def test_eq_workflow_plumed_pt():
    print(flush=True)
    pdb_in = "tests/data/pdb/gt_wob_solv.pdb"
    pdb_clean = "tests/data/pdb/gt_wob_solv_clean.pdb"
    # Remove the waters and ions
    nqe.remove_residues_in_pdb(pdb_in, pdb_clean, ['WAT', 'HOH', 'Na+', 'Cl-'])

    pdb = app.PDBFile(pdb_clean)
    potential = MLPotential('mace-off23-small')
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Equilibration workflow
    nqe.run_openmm_relaxation_simple(modeller, potential, platform_name='CUDA')
    # nqe.run_openmm_heating(modeller, potential, platform_name='CUDA')
    # nqe.run_openmm_npt(modeller, potential, platform_name='CUDA')

    # Plumed MD
    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    plumed = """# PLUMED input file for metadynamics on proton transfer in guanine-cytosine base pair
d1: DISTANCE ATOMS=10,24,11,23
METAD LABEL=metad ARG=d1 PACE=500 HEIGHT=1.2 SIGMA=0.1 FILE=HILLS
PRINT ARG=d1,metad.bias FILE=COLVAR STRIDE=100
"""

    # Write PLUMED script to a temporary file
    plumed_script_path = "plumed.dat"
    with open(plumed_script_path, 'w') as f:
        f.write(plumed)

    # nqe.run_openmm_prod(modeller, potential)

    # Clean up
    os.remove(pdb_clean)
    os.remove('minimized.chk')
    os.remove('minimized.log')
    os.remove('minimized.pdb')
    os.remove('minimized_steps.pdb')

    # os.remove('equilibrate.chk')
    # os.remove('equilibrate.log')
    # os.remove('equilibrate.pdb')
    # os.remove('equilibrate_steps.pdb')
    #
    # os.remove('npt_equilibrated.chk')
    # os.remove('npt_equilibrated.log')
    # os.remove('npt_equilibrated.pdb')
    # os.remove('npt_equilibrated_steps.pdb')
