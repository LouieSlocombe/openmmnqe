"""CPU regression tests for continuity between the two adQTB stages."""

import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3, openmm

import openmmnqe as nqe


class _OneParticleForceField:
    """Small deterministic system accepted by the public workflow helpers."""

    def createSystem(self, topology, **kwargs):
        system = openmm.System()
        system.addParticle(39.9 * unit.dalton)
        restraint = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        restraint.addGlobalParameter(
            "k", 100.0 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        restraint.addParticle(0, [])
        system.addForce(restraint)
        return system


def _one_particle_modeller():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("AR", chain)
    topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    return app.Modeller(topology, [Vec3(0.1, 0.0, 0.0)] * unit.nanometer)


def _adapted_friction_from_checkpoint(modeller, forcefield, checkpoint):
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.QTBIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    integrator.setSegmentLength(0.5 * unit.picosecond)
    integrator.setDefaultAdaptationRate(0.5)
    simulation = app.Simulation(
        modeller.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("CPU"),
    )
    simulation.loadCheckpoint(str(checkpoint))
    return np.asarray(integrator.getAdaptedFriction(0))


def test_adqtb_production_preserves_equilibrated_friction(tmp_path):
    modeller = _one_particle_modeller()
    forcefield = _OneParticleForceField()
    ready = tmp_path / "adqtb_ready"
    production = tmp_path / "adqtb_prod"

    nqe.run_openmm_adqtb_eq(
        modeller,
        forcefield,
        platform_name="CPU",
        n_report=1000,
        steps=1000,
        output_prefix=str(ready),
    )
    equilibrated = _adapted_friction_from_checkpoint(
        modeller, forcefield, ready.with_suffix(".chk")
    )

    nqe.run_openmm_adqtb_prod(
        modeller,
        forcefield,
        checkpoint_file=str(ready.with_suffix(".chk")),
        barostat_freq=None,
        platform_name="CPU",
        n_report=1000,
        steps=0,
        output_prefix=str(production),
    )
    continued = _adapted_friction_from_checkpoint(
        modeller, forcefield, production.with_suffix(".chk")
    )

    assert np.allclose(continued, equilibrated)
