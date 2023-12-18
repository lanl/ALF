import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Required Dependencies
install_requirements=[
    "parsl",
]

# Documentation Dependencies
doc_requirements = [
    "sphinx",
    "sphinx_rtd_theme",
]

# Extra dependencies
full_requirements = [

]

setuptools.setup(
    name="alf",
    version="0.0.1",
    author="",
    author_email="",
    python_requires=">3.9",
    install_requires=install_requirements,
    extras_require={"docs": doc_requirements, "full": full_requirements},
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
    packages=setuptools.find_packages(),
    include_package_data=True,
)

