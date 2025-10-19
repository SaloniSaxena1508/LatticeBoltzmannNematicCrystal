## ACTIVE LIQUID CRYSTAL IN A LID-DRIVEN CAVITY ##
This code simulates an active nematic liquid crystal embedded in a fluid. The entire system consists of a square cavity whose top side is moving from left to right. The goal is to study how the shear affects the ordering in the liquid crystal. The fluid satisfies the Navier-Stokes equations and the liquid crystal is characterized by a Beris-Edwards energy functional [1].
The code executes a hybrid Lattic Boltzmann and forward Euler scheme. The Beris-Edwards equations are integrated in time using a standard first order Euler scheme. The Navier-Stokes equations are integrated using a Lattice-Boltzmann method (LBM) [2] and the output of the LBM is given as an input to the Navier-Stokes code.
Several different velocity and ordering configurations are seen.

[1] Shear flow dynamics in the Beris-Edwards model of nematic liquid crystals. A. C. Murza, A. E. Teruel, A. D. Zarnescu, Proc. R. Soc. A 474, 20170673, 2017. 

[2] The Lattice Boltzmann Method, Principles and Practice. Kruger et al. Springer International Publishing 2017.
