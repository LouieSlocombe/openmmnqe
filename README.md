# OpenMM NQE

This repo is for running Molecular Dynamics (MD) simulations using OpenMM, specifically there are convenience functions
for running and analysing nuclear quantum effects such as Ring-Polymer Dynamics and adaptive Quantum Thermal Bath
approaches.

## Installation

Some dependencies (openmm-ml, openmm-plumed, geodesic_interpolate) are not installable from PyPI, so the package is
installed into a conda environment. See [build_tools/README.md](build_tools/README.md) for the full instructions and
the environment files.

## Citations

References for the methods and software used are collected in [CITATIONS.bib](CITATIONS.bib).
