# ReaxFF-nn for lammps

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
The parameter "nn yes" in "pair_style" will turn on the usage of neural network calculation for bond-order and bond energy, and by set "nn no" will use the ordinary ReaxFF potential. The usage of ReaxFF-nn make no different with ReaxFF except the "nn" parameters.

By set "nn no" will use ordinary ReaxFF potential, and make no different with old "REAXFF" folder. An example is in the example folder.