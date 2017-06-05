
IMAGE CLASSIFICATION USING MICROSOFT COGNITIVE TOOLKIT (CNTK)
==============

This is work in progress... (that said, the code is tested and works with CNTK rc1)

Functionality so far:
-	Download images from a provided list of urls
-	Splitting into train vs. test either randomly by filename, randomly by subdirectory, or using a provided list of test images
-	Dataset augmentation by adding rotated version of the training images (Note: CNTK supports other augmentation methods on-the-fly such as horizontal flipping or cropping, but not rotation)
-	(optionally) DNN refinement of a pre-trained ResNet model, with plot of the training and test accuracy as a function of the number of epochs
-	Running the DNN on all images in the dataset
-	(optionally) Training a Linear SVM on the output of the pre-trained (and possibly refined) DNN. Grid-search for the best C parameter, uses an efficient dedicated linear SVM implementation.
-	Quantitative evalutation: computation and plotting of the confusion matrix and precision/recall curve, using the test or the training set
-	Qualitative evaluation: visualization of the results
-	Active learning: UI to manually annotated more images. Images presented to the user are selected from possibly a very large dataset according to user-specfied criteria, e.g. images where the classifier is uncertain or images which are likely false positives.


PREREQUISITES
--------------

This code was tested using CNTK 2.0.0, and assumes that CNTK was installed with the (default)
Anaconda Python interpreter using the [script-driven installation](https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script). Note that the code will not run on previous CNTK versions due to breaking changes.

A dedicated GPU, while technically not being required, is however recommended for refining of the DNN. If you lack a strong GPU, don't want to
install CNTK yourself, or want to train on multiple GPUs, then consider using Azure's Data Science Virtual Machine. See the [Deep Learning
Toolkit](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning) for a 1-click deployment solution.

CNTK can be easily installed by following the instructions on the [setup page](https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script).
This will also automatically add an Anaconda Python distribution. At the time of writing, the default python version is 3.5 64bit.

Anaconda comes with many packages already pre-installed. The only missing packages are opencv, scikit-learn, and Pillow. These can be installed
easily using *pip* by opening a command prompt and running:
````bash
C:/local/CNTK-2-0-rc1/cntk/Scripts/cntkpy35.bat #activate CNTK's python environment
cd resources/python35_64bit_requirements/
pip.exe install -r requirements.txt
````

In the code snippet above, we assumed that the CNTK root directory is  *C:/local/CNTK-2-0-rc1/*. The opencv python wheel was originally downloaded
from this [page](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

Troubleshooting:
- The error "Batch normalization training on CPU is not yet implemented" can be caused when installing the CPU-only version of CNTK. In such cases,
try the GPU version, even if your system does not have a GPU installed.



FOLDER STRUCTURE
--------------

|Folder| Description
|---|---
|/|				                             Root directory
|/data/|			                         Directory containing the image dataset(s)
|/data/fashionTexture/|			             Upper body clothing texture dataset
|/resources/|		                         Directory containing all provided resources
|/resources/cntk/|                           Pre-trained ResNet model
|/resources/libraries/|                      Python library funtions
|/resources/python34_64_bit_requirements/|   Python wheels and requirements file for 64bit Python version 3.4
|/resources/python35_64_bit_requirements/|   Python wheels and requirements file for 64bit Python version 3.5

All scripts are located in the root directory.
