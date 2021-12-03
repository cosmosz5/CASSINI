# Welcome to CASSINI - A project to develop image reconstruction tools based con Compressed sensing an Neural Networks 

<p align="center">
  <img width="800" height="400" src="im/disk1.png">
</p>

## The project 
<p align="justify">
Interferometry delivers the highest angular resolution in Astronomy. Since the 1960s, it is being used extensively in radio astronomy and, since about a couple of decades, it has become an important player in infrared astronomy. However, infrared interferometry is restricted to sparse arrays of only a few telescopes. While imaging is arguably the most intuitive way to analyze interferometric data, recovering images from sparsely sampled visibilities is challenging. 
</p>

<p align="justify">
<strong>Which is the impact of the sparse u-v coverage on imaging?</strong>
The sparse u-v coverage obtained with infrared arrays requieres complementary “a-priori” information to recover an image. From signal theory, interferometric imaging is an “ill-possed” problem, even monochromatic images as small as 128x128 pixels requiere at least 16384 visibilities to obtain an independent solution. This number of data points is virtually imposible to obtain, given the current number of telescopes that forms infrared arrays. For example, the Very Large Telescope Interferometer (VLTI) can only provide up to 6 visibility points (per wavelength) per snapshot. Therefore, typically, there are more unknowns (or pixel values) than equations (u-v data) to solve the imaging problem. This constraint on the number of visibility points could be relaxed, if we consider that the pixels in astronomical images are not independent because we are observing highly structured morphologies. We can take advantage of the compressibility of the brightness distribution in the Fourier space and, hence, to use considerably less number of visibility points for retrieving an image. However, we would still require assumptions on the target’s morphology to achieve a reliable imaging solution. 
</p>

<p align="justify">
<strong>How does the lack of full-phase information influence imaging?</strong>
At infrared wavelengths, the atmosphere plays an important role for interferometric observations. The millisecond coherence time and the photon starved regime of the observations make virtually imposible to retrieve reliable Fourier phases as interferometric observables. In contrast with radio observations, the argument of the bispectrum (often called closure phases) of the visibilities is used for retrieving information on the centro-symmetric asymmetries of the source. One of the most important limitation of this observable is that it is shift-invariable to the position of the source in the pixel grid. This means that the image of a source could be formed at any position in the pixel grid and the resultant closure phases will be the same. This implies that, while relative astrometry is retrieved, the absolute astrometry of the imaged object lost. The limited phase information does not allow us to use direct Fourier inversion techniques (such as CLEAN) to recover images. In contrast, regularized minimization algorithms over the pixel values are required. 
</p>

<p align="justify">
<strong>This project aims at investigating new algorithms for interferometric image reconstruction based on the theory of Compressed Sensing and the novel implementation of compressibility of a signal trhough  Neural Networks to retrieve more reliable interferometric images from sparse arrays.</strong>
</p>


## Project layout

    CASSINI    # The main directory of the project
    SAMPip/ # This directory contains the software to perform data reduction of Aperture Masking Data
    CS_IM/ # This directory contains the software to image reconstruction of aperture masking data based on Compressed sensing 
    CS_PCA/ # This directory contains the software tools to perform interferometric
    CS_NN/ # This directory contains the software to perform image reconstruction based on Neural Networks (still under construction) 
    
## Installation 

At this stage, there is no need for specific installation of the code. Eash sub-package is self-contained. To use each one of them, it is as simple as clone the main repository folder with the following command: 

``` bash
>> git clone https://github.com/cosmosz5/CASSINI.git
```

However, each sub-package made use of the different Python modules available in the community. It is necessary that the user install them separately. Below, there is a description of the different requirements for each sub-module:

## Requirements 
(all the CASSINI codes wowrk with Python 3.x)

### Python modules for CASSINI/CS_PCA

The main module to run the code is <mark>PCA_im.py</mark>. This script defines the input parameters and invoke the necesary code to run the PCA analysis. The user can modify this script to adapt it to his/her necessities. The user can run it on the Terminal by typing: 

``` bash
>> python PCA_im.py
```



Required packages installed:

``` bash
numpy
astropy
cvxpy
sklearn
skimage
pylops
```

### Python modules for CASSINI/SAMpip

The main module to run the code is <mark>test.py</mark>. This script defines the input parameters and invoke the necesary code to run the Aperture Masking data reduction. The user can modify this script to adapt it to his/her necessities. The user can run it on the Terminal by typing: 

``` bash
>> python test.py
``` 

Required packages installed:

``` bash
numpy 
astropy
itertools
skimage
matplotlib
pandas
datetime
scipy
image_registration
```
At the moment, <mark>test.py</mark> finishes in a debug mode. To get out of it, the user should type "q" in the terminal


### Python modules for CASSINI/LASSO

The main module to run the code is <mark>CS_JWST_v1.py</mark>. This script defines the input parameters and invoke the necesary code to run the compressed sensins imaging. The user can modify this script to adapt it to his/her necessities. The user can run it on the Terminal by typing: 

``` bash
>> python CS_JWST_v1.py
``` 

Required packages installed:

``` bash
numpy
astropy
matplotlib
pylab
sklearn
```
At the moment, <mark>CS_JWST_v1.py</mark> finishes in a debug mode. To get out of it, the user should type "q" in the terminal

# DISCLAIMER

This project has been developed with funding from the UNAM PAPIIT project IA 101220 and from the Mexico's National Council of Science and Technology (CONACyT) “Ciencia de Frontera” project 263975. All the scripts that compose CASSINI are open source under GNU License. For enquiries and/or contributions please  contact <mailto:joelsb@astro.unam.mx>


