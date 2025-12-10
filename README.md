# Does Knowledge Actually Help?
Does using PDE-based knowledge help when applying ML methods to answer questions about spatiotemporal dynamical systems ?

## Contents
- `Data Generation.ipynb`: code to generate both Finite Differences and Pseudospectral traces
- `Correction.ipynb`: experiments to learn the correction term to add to the FD to get to PS
- `DataDrivenPINN`: currently only implements classic PINN to replace FD. Later on, it will become data-informed as well, to guide the learning
- `utils.py`: curiously useful functions