# ReaxFF-nn for LAMMPS
*ReaxFF-nn* stand for Reactive Force Field with neural networks.

Through this package can use ReaxFF-nn with LAMMPS simulator. Replace the "REAXFF" folder in "lammps/src/" directory with folder in this package and compile LAMMPS with commond like:

```shell
make yes-reaxff
make serial
```
then you will obtain the executable file name 'lmp_serial', and can be executed by command:
```shell
./lmp_seiral <in.lammps>out 
```
or execute it parallely if you compile LAMMPS with openmpi library:
```shell
nohup mpirun -n 8 ./lmp_mpi <in.lammps>out 2>&1 &  
```
To use ReaxFF-nn, you should write the pair parameters like:

```shell
pair_style     reaxff control.nn.v1 nn yes checkqeq yes
pair_coeff     * * ffield.nn.CHNO.v1 C H N O
```
and the others option in usage are same with ReaxFF, please reffer to ReaxFF examples for further usage.

"ffield.nn.CHNO.v1" is the parameter file that can be trained by ["I-ReaxFF" package](https://github.com/fenggo/I-ReaxFF). The "I-ReaxFF" package can be installed by command:
```shell
pip install irff
```
or build from source.

The parameter "nn yes" in "pair_style" will turn on the usage of neural network calculation for bond-order and bond energy, and by set "nn no" will use the ordinary ReaxFF potential. The usage of ReaxFF-nn make no different with ReaxFF except the "nn" parameters.

By set "nn no" will use ordinary ReaxFF potential, and make no different with old "REAXFF" folder. An example is given in the "example" folder.

* Refference:
1. Feng Guo et.al., Intelligent-ReaxFF: Evaluating the reactive force field parameters with machine learning, Computational Materials Science 172, 109393, 2020.

2. Feng Guo et.al., ReaxFF-MPNN machine learning potential: a combination of reactive force field and message passing neural networks,Physical Chemistry Chemical Physics, 23, 19457-19464, 2021.

3. Feng Guo et.al., ReaxFF-nn: A Reactive Machine Learning Potential in GULP and the Applications in the Thermal Conductivity Calculations of Carbon Nanostructures (Submitted, preprint: doi:10.21203/rs.3.rs-3133294/v1)

* **This package do not affect the use of ordinary ReaxFF potential!**