# Cancer-Prediction
Head-Neck cancer prediction by radiomics feature extractions using python and MATLAB

This package includes files for implementation of "Radiomics strategies for risk assessment of tumour failure in head-and-neck cancer", doi:10.1038/s41598-017-10371-5.

The package is mainly divided into 2 parts based on the environment the following scripts are compatible with (python and MATLAB).

The project is compatible with Python2.7.x and Matlab R2018b

To install the required libraries of python, run ```pip install -r requirement.txt``` from the ```python``` directory.


### Steps

Run ```./python/save2mat.py``` to save ROIs and Mask in ```.mat``` format.

Run ```./MATLAB/Master_script.m``` to save extracted features in ```.mat``` format.

Run ```./python/Feature_extract.py``` to save reduced and selected feature datasets in pickle format.
