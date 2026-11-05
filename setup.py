import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Required Dependencies
install_requirements=[
    "parsl",
    "ase",
    "matplotlib",
    "h5py",
    "pandas",
    "numpy",
]

# Documentation Dependencies
doc_requirements = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
]

# Extra dependencies
full_requirements = [
    "pyscf",
    "hippynn",
]

test_requirements = [
    "pytest",
]

# Batched MLMD with driver.engine "torchsim" (sampler worker environment; Python 3.12+).
# Install the CPU, CUDA, or ROCm torch build for the target hardware first.
torchsim_requirements = [
    "torch-sim-atomistic>=0.6.2",
]

# Qualified separately from TorchSim; sampler workers need Python 3.11+.
alchemi_requirements = ["nvalchemi-toolkit==0.3.0"]

setuptools.setup(
    name="alf",
    version="0.1.0",
    author="",
    author_email="",
    python_requires=">3.9",
    install_requires=install_requirements,
    extras_require={"docs": doc_requirements, "full": full_requirements, "tests": test_requirements,
                    "torchsim": torchsim_requirements, "alchemi": alchemi_requirements, "topology": ["rdkit"]},
    license="BSD 3-Clause License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    description="Active Learning Framework",
    long_description=long_description,
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
)
