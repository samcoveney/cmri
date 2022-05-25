# README

Collection of routines for post-processing Cardiac MRI.


## Installation

Needs the following:
* [SimpleElastiX](https://simpleelastix.readthedocs.io/GettingStarted.html) (installed with Python bindings)
* [DIPY](https://dipy.org/documentation/1.5.0/installation/)

It is best to install these in a dedicated Python environment. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a good choice for creating these environments. 

This package CMRI can be installed from the root directory using:

```
python setup.py install
```

or if you are developing the package: 

```
python setup.py develop
```


## Data format

Data is currently required to be in a NIfTI format (either .nii or .nii.gz). To convert DICOM files into a NIfTI file

```python

import dicom2nifti as dn

# set names of directories dicom_dir and output_dir

dn.convert_directory(dicom_dir, output_dir, compression=True, reorient=False)

```

For multi-slice data, the result should be a 4D NIfTI file with RAS orientiation, with the acquisitions stored in the last dimension.


## Usage

Designed to be used with scripts, use `cmri_script_name --help` to view command line arguments.

The following scripts should be available from the command line:

* `cmri_reg_series` - register a series of 2D slices
* `cmri_denoise` - denoise and identify outliers
* `cmri_fit_tensors` - fit tensors and calculate related quantities

