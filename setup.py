from setuptools import setup, find_packages

setup(name='pet_cli', version='0.0.1', packages=find_packages(),
      install_requires=['docker',
                        'numpy',
                        'scipy',
                        'numba',
                        'pandas',
                        'nibabel',
                        'antspyx',
                        'SimpleITK',
                        'matplotlib',
                        'sphinx',
                        'pydata-sphinx-theme',
                        'bids_validator'],
      entry_points={'console_scripts': ['pet-cli-preproc = pet_cli.cli_preproc:main',
                                        'pet-cli-bids = pet_cli.cli_bids:main',
                                        'pet-cli-tac-interpolate = pet_cli.cli_tac_interpolation:main',
                                        'pet-cli-graph-plot = pet_cli.cli_graphical_plots:main',
                                        'pet-cli-graph-analysis = pet_cli.cli_graphical_analysis:main',
                                        'pet-cli-parametric-image = pet_cli.cli_parametric_images:main',
                                        'pet-cli-tcm-fit = pet_cli.cli_tac_fitting:main',
                                        'pet-cli-rtms = pet_cli.cli_reference_tissue_models:main',
                                        'pet-cli-pvc = pet_cli.cli_partial_volume_correction:main',
                                        'pet-cli-idif = pet_cli.cli_idif:main']
                    }
      )
