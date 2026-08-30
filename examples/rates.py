"""Ring-polymer rate dynamics: recrossing, rate constants, and spectra."""

from __future__ import annotations

import argparse
import os
import re

import forcefill as ff
import numpy as np
import openmm.app as app
import openmm.unit as unit
import reactiontools as rt
from openmmml import MLPotential

import openmmnqe as nqe

device = "CUDA"
BASE_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")


def ligand_forcefield(input_pdb: str) -> tuple[app.Modeller, app.ForceField]:
    """Parameterise a ligand PDB and return the matching modeller and force field."""
    result = ff.build_forcefield_xml(input_pdb, "ligands.xml",
                                     base_forcefield=BASE_FORCEFIELD)
    assert not result.skipped, f"forcefill skipped residues: {result.skipped}"
    extra = [] if result.forcefield_xml is None else [result.forcefield_xml]
    pdb = app.PDBFile(input_pdb)
    return (app.Modeller(pdb.topology, pdb.positions),
            app.ForceField(*BASE_FORCEFIELD, *extra))


def run_malonaldehyde_rate() -> None:
    """Assemble k = kappa * k_QTST for the malonaldehyde proton transfer.

    Four stages: metadynamics gives the free-energy surface and the barrier
    position, a stiff PLUMED restraint holds a thermostatted run at that
    barrier while snapshots of every bead are harvested, unbiased
    thermostat-off children shot from the snapshots give the transmission
    coefficient, and the rate is assembled from the two. Rerun with
    ``deuterate=True`` on every stage (and ``deuterate_option='all'``) and
    the ratio of the two rates is the H/D kinetic isotope effect.
    """
    print(flush=True)
    temperature = 300.0 * unit.kelvin
    n_beads = 16

    modeller, forcefield = ligand_forcefield("tests/data/pdb/malonaldehyde.pdb")
    idx = nqe.atom_indices_from_vmd_picks(
        modeller, ["LIG1:O2", "LIG1:H5", "LIG1:O1"],
    )
    # A classical force field cannot break the O-H bond, so the transferring
    # proton needs a reactive potential: run the whole molecule on MACE.
    potential = MLPotential("mace-off23-small")
    ml_atoms = [atom.index for atom in modeller.topology.atoms()]

    # --- 1. The free-energy surface, from RPMD metadynamics.
    nqe.run_openmm_rpmd_equilibration(modeller, forcefield, n_beads=n_beads,
                                      temperature=temperature,
                                      potential=potential, ml_idx=ml_atoms,
                                      platform_name=device)

    plumed_input, sum_hills_input = rt.plumed_input_1pt(modeller, idx,
                                                        temperature)
    with open("plumed.dat", "w") as handle:
        handle.write(plumed_input)
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=n_beads,
                             temperature=temperature,
                             plumed_script_path="plumed.dat",
                             barostat_freq=None,
                             steps=200_000,
                             potential=potential, ml_idx=ml_atoms,
                             platform_name=device)
    os.system(sum_hills_input)

    # The proton starts donor-bound (pt_cv near +1) and transfers to -1;
    # the dividing surface is the barrier top between the two basins.
    summary = rt.summarise_fes("fes.dat", basin_a=(0.5, 1.05),
                               basin_b=(-1.05, -0.5))
    s_dagger = summary.barrier_position
    print(summary)

    # --- 2. Harvest full-bead snapshots under a stiff restraint at the
    # barrier. reactiontools has no fixed-restraint builder yet, so reuse
    # the CV and wall definitions of the metadynamics script and swap the
    # bias for a RESTRAINT pinned at s_dagger.
    cv_block = plumed_input.split("# Metadynamics")[0]
    with open("restraint.dat", "w") as handle:
        handle.write(
            f"{cv_block}\n"
            f"pin:        RESTRAINT ARG=pt_cv AT={s_dagger:.6f} KAPPA=10000.0\n"
        )
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=n_beads,
                             temperature=temperature,
                             plumed_script_path="restraint.dat",
                             checkpoint_file="rpmd_ready.chk",
                             output_prefix="pinned",
                             barostat_freq=None,
                             steps=20_000,
                             potential=potential, ml_idx=ml_atoms,
                             platform_name=device)
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=n_beads,
                             temperature=temperature,
                             plumed_script_path="restraint.dat",
                             checkpoint_file="pinned.chk",
                             output_prefix="harvest",
                             barostat_freq=None,
                             steps=100_000,
                             snapshot_interval=1_000,
                             potential=potential, ml_idx=ml_atoms,
                             platform_name=device)

    # --- 3. Shoot unbiased thermostat-off children from every snapshot.
    # The Python collective variable must be the one PLUMED biased:
    # coordination(donor-H) - coordination(acceptor-H), with the R_0 the
    # script actually carries.
    r_0 = float(re.search(r"R_0=([0-9eE.+-]+)", plumed_input).group(1))
    donor, hydrogen, acceptor = idx

    def pt_cv(positions_nm: np.ndarray) -> float:
        r_dh = float(np.linalg.norm(positions_nm[donor] - positions_nm[hydrogen]))
        r_ah = float(np.linalg.norm(positions_nm[acceptor] - positions_nm[hydrogen]))
        return rt.switching_value(r_dh, r_0) - rt.switching_value(r_ah, r_0)

    snapshots = sorted(nqe.list_files_with_pattern(".", "harvest_snapshot_*.npz"))
    nqe.run_openmm_rpmd_recrossing(modeller, forcefield, snapshots, pt_cv,
                                   n_beads=n_beads,
                                   temperature=temperature,
                                   n_children=10,
                                   n_steps=400,
                                   record_interval=5,
                                   potential=potential, ml_idx=ml_atoms,
                                   platform_name=device,
                                   seed=1)

    logs = sorted(nqe.list_files_with_pattern(".", "rpmd_recrossing_recrossing_*.log"))
    result = nqe.transmission_coefficient(logs, s_dagger=s_dagger)
    nqe.plot_transmission_coefficient(logs, s_dagger=s_dagger,
                                      filename="kappa.png")
    print(f"kappa = {result.plateau:.3f} +/- {result.plateau_stderr:.3f} "
          f"(initial CV spread {result.s0_std:.3f})")

    # --- 4. Assemble the rate from kappa and the surface.
    fes = rt.as_fes("fes.dat")
    order = np.argsort(fes.cvs[0])
    rate = nqe.rpmd_rate(result,
                         fes.cvs[0][order],
                         fes.energy[order],
                         temperature=temperature,
                         s_dagger=s_dagger,
                         reactant_window=(0.5, 1.05))
    print(f"k_QTST = {rate.qtst_rate:.3e} 1/ps, "
          f"k = {rate.rate:.3e} +/- {rate.rate_stderr:.3e} 1/ps")


def run_rpmd_energy_conservation() -> None:
    """Run thermostat-off RPMD and verify its conserved quantity held."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller, forcefield, n_beads=16,
                                      platform_name=device)
    # apply_thermostat=False is microcanonical ring-polymer dynamics: the
    # ensemble every RPMD time-correlation observable is defined in. With
    # no thermostat to absorb integration error, run a smaller time step
    # and check the ring-polymer Hamiltonian afterwards.
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=16,
                             apply_thermostat=False,
                             barostat_freq=None,
                             time_step=0.5 * unit.femtosecond,
                             steps=20_000,
                             n_report=100,
                             output_prefix="rpmd_nve",
                             platform_name=device)

    verdict = nqe.rpmd_energy_conservation("rpmd_nve_thermo.log",
                                           temperature=300.0,
                                           discard=0.1)
    state = "conserved" if verdict.conserved else "DRIFTING"
    print(f"E_ring {state}: drift {verdict.drift_rate:.3e} kJ/mol/ps "
          f"({verdict.drift_per_ps_over_kbt:.2e} kBT/ps), "
          f"fluctuation {verdict.fluctuation:.3f} kJ/mol")

    nqe.remove_file_pattern("rpmd_ready*")
    nqe.remove_file_pattern("rpmd_nve*")


def run_rpmd_spectrum() -> None:
    """Record centroid velocities thermostat-off and take a vibrational spectrum."""
    print(flush=True)
    pdb = app.PDBFile("tests/data/pdb/input_aaa.pdb")
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens()

    nqe.run_openmm_rpmd_equilibration(modeller, forcefield, n_beads=16,
                                      platform_name=device)
    # Record every 4 steps of 0.5 fs: a 2 fs sampling interval resolves
    # frequencies up to ~8300 cm^-1, comfortably past the X-H stretches.
    nqe.run_openmm_rpmd_prod(modeller, forcefield, n_beads=16,
                             apply_thermostat=False,
                             barostat_freq=None,
                             time_step=0.5 * unit.femtosecond,
                             steps=100_000,
                             n_report=10_000,
                             velocity_record_interval=4,
                             output_prefix="rpmd_nve",
                             platform_name=device)

    frequencies, intensities = nqe.rpmd_vibrational_spectrum(
        "rpmd_nve_velocities.npz", max_time=2.0,
    )
    np.savetxt("spectrum.dat", np.column_stack([frequencies, intensities]),
               header="frequency(cm^-1)\tintensity(Da nm^2/ps)")
    strongest = frequencies[np.argsort(intensities)[-5:]]
    print("five strongest bands (cm^-1):",
          " ".join(f"{value:.0f}" for value in sorted(strongest)))

    nqe.remove_file_pattern("rpmd_ready*")
    nqe.remove_file_pattern("rpmd_nve*")


EXAMPLES = {
    name.removeprefix("run_"): function
    for name, function in list(globals().items())
    if name.startswith("run_") and callable(function)
}


def main() -> None:
    """Run whichever example is named on the command line."""
    global device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--platform", default=device)
    args = parser.parse_args()

    device = args.platform
    EXAMPLES[args.example]()


if __name__ == "__main__":
    main()
