# This repository includes software to recover infrared interferometric images based on Compressed Sensing


## 1 Compressed Sensing
<p align="justify">
Compressed Sensing (CS) allows us to recover a signal with less samples that the ones established from the Nyquist/Shannon theorem (see e.g. <sup>9-11</sup>). For the technique to work, the signal must be sparse and compressible on a given basis. It means that the signal can be represented by a linear combination of functions with a small number of non-zero coefficients. In CS, a set of measurements, <strong>y</strong>, of a given signal, <strong>x</strong>, can be encoded by a multiplication of the matrices <strong>&Phi;</strong>, <strong>&Psi;</strong>, and the sparse vector <strong>&alpha;</strong>. <strong>&Psi;</strong> is the transformation basis where the full signal, <strong>x</strong>, is sparse, and only a few coefficients in the vector <strong>&alpha;</strong> are non-zero. <strong>&Phi;</strong> is, thus, the system of measurements under which the data are taken. For a visual representation of the matrices involved in CS see Fig. 1. It is important to remark that the number of measurements in <strong>y</strong> is considerably smaller than the number of features/columns in  in <strong>&Psi;</strong>, therefore, the inverse problem to find <strong>&alpha;</strong> is "ill-posed". CS establishes that if the product &Theta; = <strong>&Phi;&Psi;</strong> satisfies the Restricted Isometry Property (RIP)<sup>10, 12</sup>, we will be able to recover the signal from the sub-sampled measurements. Therefore, compressed Sensing offers us a framework to solve the "ill-posed" inverse problem by a regularized optimization, using as prior the sparsity of &alpha; and/or the degree of compressibility of the signal. 
</p>

<p align="justify">
Interferometric data are ideal to use Compressed Sensing for two reasons: (i) the data are a series of semi- independent measurements which provide the incoherent sampling that is needed; (ii) the interferometric data are measurements of structured images, it means that the images are highly compressible.
The role of CS for inteferometric imaging has gain importance in the recent years. Particularly, there has been new developments in Radio Astronomy. For example PURIFY<sup>13</sup>, shows how reconstruction algorithms based on Compressed Sensing outperforms CLEAN and its variants such as MS-CLEAN and ASP-CLEAN. It is interesting to mention that this work discusses the increment in processing speed gained with Compressed Sensing over other methods. More recently, <sup>14</sup> uses CS for imaging real Very Large Array (VLA) data; and  <sup>15</sup> highlights the use of CS for dimensionality reduction applied to radio inferferometric data. Additional works on CS applied to astronomical imaging include <sup>16, 17</sup> and <sup>18</sup>.
</p>

<p align="justify">
This repository includes code to recover infrared interferometric images using CS from simulated Aperture Masking data. The Aperture Masking data is simulated as expected to be recorded by the near-infrared imager NIRISS on-board of the James Webb Space Telescope (JWST).

</p>

  
## 2. James-Webb Space Telescope Simulations
<p align="justify">
NIRISS (Near Infrared Imager and Slitless Spectrograph) is an infrared (band-pass = 0.8 - 5μm) high-resolution camera which allows us to observe an object using Fizeau interferometry in the form of Sparse Aperture Masking (SAM). SAM is a technique which allows us to transform a telescope into an interferometer by placing a non-redundant mask with several pin-holes in the pupil plane of the telescope<sup>21</sup>. Therefore, at the image plane an interferogram is  formed  (see  Fig.   2). From  the  interferograms,  interferometric  observables  (Fourier  visibilities  and  phases, squared  visibilities  and  closure  phases)  are  extracted.   The  non-redundant  mask  on-board  of  NIRISS  has 7 pinholes, which produces 21 visibilities and 35 closure phases per snapshot.  For the simulations reported, we fitted the fringes directly in the image plane using a model of the mask geometry and filter bandwidth. From this model the observables were computed using a single-value decomposition (SVD) algorithm. This method is similar to the one presented by <sup>22</sup>.  To evaluate the validity of our algorithm, we compared the observables extracted with the ones obtained with ImPlaneIA<sup>23</sup> and AmiCal, finding similar results.
</p> 




# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
