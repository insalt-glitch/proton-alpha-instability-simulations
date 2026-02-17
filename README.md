# Kinetic Simulations of the Proton-Alpha Instability in Collisionless Shocks
This project contains the all the source code of the data analysis in my Master's thesis. The report is openly available [here](TBD) and the data used for the analysis can be found [here](TBD). If you have any questions about the project you can contact [me](mailto:nils.n.mueller@gmail.com).

# Requirements
For the development I used [Python 3.12](https://www.python.org/downloads/release/python-3120/) and all the required packages can be found in `requirements.txt`. You can install all packages using `pip install -r requirements.txt`.
# How to use
All plots and videos can recreated from `overview.ipynb`. This relies on the fact that correct paths are specified in `basic/paths`. the file `plots/settings.py` can be used configure the plot quality and output-formats. 

Additionally, there are separate tools in `tools/`. These are used primarily to convert the output files from the [EPOCH code](https://epochpic.github.io/), which are in the SDF-format, to the HDF5-format. First run `tools/sdf2h5.py` on the SDF-files; they will be converted to HDF5 while the original files are optionally archived. Subsequently, run `tools/h5vstack.py` on the directory which contains the individual HDF5 files. For the publication, the files can be directly prepared in a reproducable way. This takes roughly 300 - 400 GB of storage space (unless you manually delete files along the way). Follow these steps:

1. Run `bash tools/extract_files.sh <download-directory> <target-directory>`. The SDF-files from the PIC simulations will then be extracted into this directory.
2. In `basic/paths.py` change the RESULTS_FOLDER variable to the directory of you're choosing.
3. Install/prepare a python3-environment with the dependencies listed in `pyproject.toml`. The following script is using the [pixi](https://pixi.prefix.dev) environment-manager. In this case, you just enter the `pixi init` command.
4. Run `bash tools/convert_files.sh <target-directory>` in your python-environment. For pixi, use `pixi run ...` or start the environment with `pixi shell` before running the command.
5. Finally, you can run `python -m plots.publication` to reproduce all plots and auxillary information from the manuscript.

Note, the files `u_alpha_dispersion.py` and `density_ratio_dispersion.py` are recreated in Python from [MATLAB (Daniel Graham)](https://github.com/danbgraham/ShocksIonIonInstability). These are used compute find numerical solutions to the dispersion relation of the plasma.

# Reproduce manuscript figures
**BE AWARE:** The simulation output-files when extracted and converted take in total roughly 400 GB of disk space!

For clarity in a separate section, the information conndensed from ***How to use***.  To reproduce the figures, a python environment is required that contains the packages specified in `pyproject.toml`. Here, I assume you are using [pixi](https://pixi.prefix.dev):
1. Download all `tar`-archives from the zenodo repository.
2. Install [pixi](https://pixi.prefix.dev)
3. In `basic/paths.py` change the `RESULTS_FOLDER` variable to the directory of you're choosing. In the following, this directory is denoted as `<results-directory>`
4. Exectue the following commands:
```(bash)
cd <this-directory>
pixi init
pixi shell
bash tools/extract_files.sh <download-directory> <results-directory>
bash tools/convert_files.sh <results-directory>
python -m plots.publication
```



