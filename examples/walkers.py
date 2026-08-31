"""Independent OPES walkers run as parallel processes, merged by reweighting."""

from __future__ import annotations

import openmm.app as app
import openmm.unit as unit
import reactiontools as rt

import openmmnqe as nqe


def main() -> None:
    """Equilibrate once, fan out four OPES walkers, merge them into one FES."""
    n_walkers = 4
    n_steps = 100_000
    temperature = 300.0 * unit.kelvin
    kbt = rt.thermal_energy(temperature)

    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField('amber14-all.xml',
                                'amber14/tip3pfb.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()
    modeller.addSolvent(forcefield,
                        padding=1.5 * unit.nanometer,
                        boxShape='cube')
    nqe.center_in_box(modeller)

    # One equilibration serves every walker; only production fans out.
    nqe.run_openmm_relaxation_simple(modeller, forcefield)

    pdb = app.PDBFile("minimized.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_heating(modeller, forcefield)

    pdb = app.PDBFile("equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nqe.run_openmm_npt(modeller, forcefield)

    pdb = app.PDBFile("npt_equilibrate.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)

    idx = nqe.atom_indices_from_vmd_picks(modeller, ['ALA1:C', 'ALA2:N', 'ALA2:CA', 'ALA2:C'])
    idx_str = ",".join(str(i + 1) for i in idx)
    plumed_input = f"""
phi: TORSION ATOMS={idx_str}
metad: OPES_METAD ARG=phi PACE=500 BARRIER=4.0 SIGMA=0.2 TEMP={temperature.value_in_unit(unit.kelvin)} STATE_WFILE=STATE STATE_WSTRIDE=500
PRINT STRIDE=200 ARG=phi,metad.bias FILE=COLVAR
"""

    # OPES_METAD shares bias between walkers only over MPI, which this build
    # does not have, so the walkers are independent: the same input, disjoint
    # derived seeds, and one directory and process each. max_workers=1 would
    # serialise them on a single GPU; per_walker_kwargs would hand each its
    # own starting Modeller.
    ensemble = nqe.run_openmm_walkers(
        nqe.run_openmm_prod,
        n_walkers,
        dict(modeller=modeller, forcefield=forcefield, steps=n_steps),
        plumed_input=plumed_input,
        seed=2024,
        resume=True,
    )

    # Every sample carries its own walker's bias, so the merged COLVAR
    # reweights as one estimate. Unsorted keeps each walker contiguous, and
    # blocks= then puts the error bars on the cross-walker scatter.
    rt.combine_colvar_files(
        [f"{directory}/COLVAR" for directory in ensemble.directories],
        sort_by_time=False,
    )
    rt.run_opes_reweighting(sigma=0.1,
                            kt=kbt,
                            cv="phi",
                            grid_min=-3.14,
                            grid_max=3.14,
                            grid_bin=100,
                            blocks=n_walkers)
    rt.plot_plumed_fes("fes.dat", show=True)

    nqe.remove_file_pattern('minimized*')
    nqe.remove_file_pattern('equilibrate*')
    nqe.remove_file_pattern('npt_equilibrate*')
    for directory in ensemble.directories:
        nqe.remove_directory(directory)
    nqe.remove_file('COLVAR')
    nqe.remove_file('fes.dat')


if __name__ == "__main__":
    main()
