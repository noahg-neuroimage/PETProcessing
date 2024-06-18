from setuptools import setup, find_packages

setup(name='petpal', version='0.0.1', packages=find_packages(),
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
      entry_points={'console_scripts': ['petpal-preproc = petpal.cli.cli_preproc:main',
                                        'petpal-bids = petpal.cli.cli_bids:main',
                                        'petpal-tac-interpolate = petpal.cli.cli_tac_interpolation:main',
                                        'petpal-graph-plot = petpal.cli.cli_graphical_plots:main',
                                        'petpal-graph-analysis = petpal.cli.cli_graphical_analysis:main',
                                        'petpal-parametric-image = petpal.cli.cli_parametric_images:main',
                                        'petpal-tcm-fit = petpal.cli.cli_tac_fitting:main',
                                        'petpal-rtms = petpal.cli.cli_reference_tissue_models:main'], }, )
