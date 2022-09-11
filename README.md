 Incorporated Research Institutions for Seismology (IRIS)\
 Data Management Center (DMC)\
 Data Products Team
 
 Earth Model Collaboration (EMC)\
 EMC Model Explorer - an  EMC Notebook

 2022-09-09

------------------------------------------------------------------------------------------------------------------------

 DESCRIPTION:

This Jupyter Notebook reads an [EMC-compatible Earth model file](http://ds.iris.edu/ds/products/emc-earthmodels/) in the geographic coordinate system, and allows the user to explore its contents by: 

* Displaying the model metadata in either [netCDF](https://www.unidata.ucar.edu/software/netcdf/) or [GeoCSV](http://geows.ds.iris.edu/documents/GeoCSV.pdf) formats
* Creating horizontal slices (maps), vertical slices (cross-sections), and velocity profiles (line plots vs depth) from model data
* Saving the generated plots and their corresponding data to a local directory
* Outputing the entire netCDF file, or a selected portion of it in GeoCSV format

 CONTENT:

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
       A directory of sample EMC model netCDF files.

    CHANGES.txt
      History of changes to this package.

    INSTALL.txt
       The installation notes

    README.md
       The package README file 


CITATION:

To cite the use of this software reference:

Trabant, C., A. R. Hutko, M. Bahavar, R. Karstens, T. Ahern, and R. Aster (2012), Data Products at the IRIS DMC: \
Stepping Stones for Research and Other Applications, Seismological Research Letters, 83(5), 846–854, \
https://doi.org/10.1785/0220120032.\


Or cite the following DOI:

    doi:10.17611/dp/emc.notebooks.1

 
 COMMENTS/QUESTIONS:

    Please contact manochehr.bahavar@iris.edu


