# Positron Emissions Tomography Processing and Analysis Library (PETPAL)

A comprehensive 4D-PET analysis software suite.

## Installation

In the top-level directory (where `pyproject.toml` exists), we run the following command in the terminal:

```shell
pip install .  # Installs the package
```

## Generating Documentation

To generate the documentation in HTML using sphinx, assuming we are in the `docs/` directory and that sphinx is
installed:

```shell
make clean
make html 
```

Then, open `doc/build/html/index.html` using any browser or your IDE.
