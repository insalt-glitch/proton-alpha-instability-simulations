# Kinetic Simulations of the Proton-Alpha Instability in Collisionless Shocks
This project contains the all the source code of the data analysis in my Master's thesis. The report is openly available [here](TBD) and the data used for the analysis can be found [here](TBD). If you have any questions about the project you can contact [me](mailto:nils.n.mueller@gmail.com).

# Requirements
For the development I used [Python 3.12](https://www.python.org/downloads/release/python-3120/) and all the required packages can be found in `requirements.txt`. You can install all packages using `pip install -r requirements.txt`.
# How to use
All plots and videos can recreated from `overview.ipynb`. This relies on the fact that correct paths are specified in `basic/paths`. the file `plots/settings.py` can be used configure the plot quality and output-formats. 

Additionally, there are separate tools in `tools/`. These are used primarily to convert the output files from the [EPOCH code](https://epochpic.github.io/), which are in the SDF-format, to the HDF5-format.

Finally, the files `u_alpha_dispersion.py` and `density_ratio_dispersion.py` are recreated in Python from [MATLAB (Daniel Graham)](https://github.com/danbgraham/ShocksIonIonInstability). These are used compute find numerical solutions to the dispersion relation of the plasma.
