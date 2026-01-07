# Installation guide

This guide will help you install and set up the project on your local machine.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a compatible operating system: Windows (WSL), macOS, or Linux.
- You will need to have Python 3.8 or higher installed.
- Conda or Mamba package manager installed.

## Installation Steps

1. **Install ORCA**: Follow the instructions on the [ORCA website](https://www.faccts.de/orca/)
2. **Extract the downloaded ORCA package** `tar -xf orca-x.y.z.tar.gz`
3. **Add ORCA to your PATH**:
    - On Linux/macOS, add the following line to your `~/.bashrc` file, replacing the path with the actual path to your
      ORCA installation:
      ```bash
      export ORCA_PATH="/home/lslocomb/orca_6_1_1/orca"
      ```
    - For Windows it is recommended to use the Linux Subsystem for Windows (WSL) for better compatibility.
4. **Create a Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```
5. **Activate the Conda environment**:
    ```bash
    conda activate openmmnqe
    ```
6. **Run the files in execution** to perform your desired calculations.