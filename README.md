 
 
# Earth Model Collaboration (EMC) Model Explorer

This Jupyter Notebook reads an [EMC-compatible Earth model file](http://ds.iris.edu/ds/products/emc-earthmodels/) in the geographic coordinate system, and allows the user to explore its contents by: 

* Displaying the model metadata in either [netCDF](https://www.unidata.ucar.edu/software/netcdf/) or [GeoCSV](http://geows.ds.iris.edu/documents/GeoCSV.pdf) formats
* Creating horizontal slices (maps), vertical slices (cross-sections), and velocity profiles (line plots vs depth) from the model data
* Saving the generated plots and their corresponding data to a local directory
* Outputting the entire netCDF file, or a selected portion of it in GeoCSV format

## Installation

Either clone the repository or download a [release](https://github.com/iris-edu/emc-notebooks/releases) and unzip/untar it.

### Requirements

* [Python](https://www.python.org/) 3
* Python modules listed in `requirements.txt`
  * Install these modules with `pip install -r requirements.txt`

This package has been tested under Python 3.10.06 on macOS 12.5.1, it may work with older Python 3 versions.

### Running the package

Notebooks can be executed a variety of ways, commonly either with JupyterLab or Jupyter Notebook (https://jupyter.org/).

Using the `jupyterlab` module install from `requirements.txt` you can simply execute the following in a terminal:

```console
jupyter-lab
```

and open the `notebooks/emc_model_explorer.ipynb` notebook. For more information visit the [Wiki page](https://github.com/iris-edu/emc-notebooks/wiki)

### Package testing

This package comes with a `samples` directory pre-loaded with a few EMC netCDF files for testing. 
Run the notebook as it is, using the "Restart and Run all" option (>>).
It should load a sample model file from the samples directory, and display information and plots for the model.

You load other model files using the **file selection widget** under the **Select a Model** section.

## Package contents

This package contains the following files:

     notebooks/
       emc_model_explorer.ipynb
           This is the EMC model explorer Jupyter Notebook. 
       bemc_utils.py
           A Python utility library used by the notebook.
     
       assets/
           An asset directory for the notebook to hold images and files used by the notebook.

     data/
       The directory where the model netCDF files are read from.

     output/
       The directory where the notebook stores its output files under.

     samples/
       A directory of sample EMC model files.

    requirements.txt
        List of the required Python modules.

    CHANGES.txt
      History of changes to this package.

    README.md
       The package README file 

## CITATION

To cite the use of this software reference:

```
Trabant, C., A. R. Hutko, M. Bahavar, R. Karstens, T. Ahern, and R. Aster (2012), Data Products at the IRIS DMC:
Stepping Stones for Research and Other Applications, Seismological Research Letters, 83(5), 846–854,
https://doi.org/10.1785/0220120032
```

Or cite the following DOI:

```
doi:10.17611/dp/emc.notebooks.1
```

## Authors

Incorporated Research Institutions for Seismology (IRIS)
Data Management Center (DMC)
Data Products Team

### Comments or questions

  Please contact manochehr.bahavar@iris.edu


## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2022 Manochehr Bahavar, IRIS Data Management Center