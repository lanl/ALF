module switch PrgEnv-cray PrgEnv-gnu
module load cuda/11.5
module load cpe-cuda
module load cray-libsci
module load cray-fftw
module load python/3.9-anaconda-2021.11
module load cmake

source activate /usr/projects/ml4chem/envs/p310-alf

#------------------parsl-ALF directory-----------------#
export PYTHONPATH="/users/vitor/ALF/:$PYTHONPATH"
export PYTHONPATH="/users/vitor/moltensalt_builder:$PYTHONPATH"
export PYTHONPATH="/usr/projects/ml4chem/envs/p310-alf-packages/PythonPath/:$PYTHONPATH"

#-------------------ANI-Tools LIB----------------------#
export PYTHONPATH="/usr/projects/ml4chem/Programs/ANI-Tools-update/lib:$PYTHONPATH"

#-------------------ASE-ANI LIB------------------------#
##export PATH="/projects/ml4chem/Programs/ASE_ANI/lib:$PATH"

#--------------------NeuroChem EXPORTS---------------------#
export LD_LIBRARY_PATH="/usr/projects/ml4chem/Programs/ANAKIN/NeuroChem/build-ch/lib/:$LD_LIBRARY_PATH"
export PYTHONPATH="/usr/projects/ml4chem/Programs/ANAKIN/NeuroChem/build-ch/lib/:$PYTHONPATH"
export PATH="/usr/projects/ml4chem/Programs/ANAKIN/NeuroChem/build-ch/bin/:$PATH"

#-------------------boost EXPORTS---------------------#
export LD_LIBRARY_PATH="/usr/projects/ml4chem/Programs/boost_1_63_0/stage/lib:$LD_LIBRARY_PATH"
export Boost_INCLUDE_DIR="/usr/projects/ml4chem/Programs/boost_1_63_0/stage/include/"

#------------------- ASE VASP exports ----------------#
export VASP_PP_PATH="/users/vitor/VASP/vasp_potentials"

#-------------------- ORCA MPI exports -----------------------#
export PATH=/usr/projects/ml4chem/Programs/openmpi-4.1.1_parsl/install/bin/:/usr/projects/w20_mlhamil/orca_5_0_1_linux_x86-64_shared_openmpi411/:$PATH
export LD_LIBRARY_PATH=/usr/projects/ml4chem/Programs/openmpi-4.1.1_parsl/install/lib/:/usr/projects/w20_mlhamil/orca_5_0_1_linux_x86-64_shared_openmpi411/:$LD_LIBRARY_PATH

