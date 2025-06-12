
# 2D/3D Multi-GPU Deconvolution of Multi-Terabyte Images


<!-- ABOUT THE PROJECT -->
## About The Project

The scripts provide capability to run 2D or 3D Richardson-Lucy deconvolution on 
TIFF images with theoretical PSF. The images can be in the order of Terabytes. Multiple GPUs can be
used for deconvolution on a cluster.


<!--Prerequisites -->
## Prerequisites for Linux
We have extensively tested it on Linux with the following versions,
* Python 3.10 in Anaconda 2023.03-1 version, which can be downloaded from here,
```
https://repo.anaconda.com/archive/
```
* CUDA-11.2.2
* CUDNN 8.1.0.77
* Other requirements:
```
pip install nibabel 
pip install pynvml
pip install scipy==1.13.0 
pip install scikit-image==0.21.0
pip install --upgrade pillow
pip install --upgrade imagecodecs
pip install tensorflow-gpu==2.10.0
pip install cupy-cuda11x
pip install cucim==23.4.1
```


## Prerequisites for Windows
See the [Powerpoint documentation](https://github.com/SNIR-NIMH/Deconvolution/blob/main/Deconvolution%20GUI%20on%20Windows.pptx).
Note that administrator access is required. The Windows GUI only uses a single GPU. For multiple GPUs, use it on Linux.


<!-- USAGE EXAMPLES -->
## Single-GPU Deconvolution on Linux
For a single-GPU deconvolution, use deconvolution3d.py
```
python deconvolution3d.py -h

usage: deconvolution3d.py [-h] --im IMAGE --psf PSF --o OUTPUT [--iter NUMITER] [--gpu GPUID] [--chunks CHUNKS [CHUNKS ...]]
                          [--float]

******************************************************************* 
Deconvolution using RAPIDSAI cuCIM (Compute Unified Device Architecture Clara IMage).
Use this code for smaller size images, e.g. size 2560x2160x1200.
For larger images, use deconvolution_cluster_prepare.py and deconvolution_cluster_run.py,
which deconvolves with parallel GPUs on a cluster.
******************************************************************* 

options:
  -h, --help            show this help message and exit

Required Arguments:
  --im IMAGE, -i IMAGE  Input 3D tif image stack, or a directory containing multiple 2D tif images.
  --psf PSF             PSF 3D tif image.
  --o OUTPUT, -o OUTPUT
                        Output image, either a .tif file or a folder where output 2D slices will be written. If the output ends
                        with .tif, it is assumed that a 3D file will be written. Dont use the .tif option if the image size is
                        too big to fit in RAM, e.g. stitched images.

Optional arguments:
  --iter NUMITER        Number of iterations, default 10.
  --gpu GPUID           GPU id to use, default 0
  --chunks CHUNKS [CHUNKS ...]
                        If this is mentioned, then the image is split into given chunks and deconvolved. Use this option if the
                        image X-Y size is too large (>2560 pixels) to fit in RAM. If this option is used, deconvolution will be
                        slower.
  --float               Use this option to save output images as FLOAT32. Default is UINT16. This is useful if the dynamic
                        range of the image is small. Note, saving as FLOAT32 images will approximately double the size of the
                        image.
```

Example:
```
python deconvolution3d.py --im A_3d_tif_stack.tif --o decon.tif --psf psf.tif   
python deconvolution3d.py --im /home/user/a_dir_with_2D_tifs/ --o  /home/user/outputimage.tif --psf psf.tif
python deconvolution3d.py --im /home/user/stitched_image_dir/ --o  /home/user/outputdir/ --psf psf.tif --chunks 20  20
```

* Number of chunks should be empirically chosen. For 3D deconvolution, more number of chunks may be required.
* There is no limit on number of chunks or the image XY dimension. An arbitrary limit of 45000x45000 pixel dimension 
  is added to Pillow's MAX_IMAGE_PIXELS. Please change L22 of deconvolution3d.py to accommodate bigger image size, e.g.,
```
Image.MAX_IMAGE_PIXELS = 45000*45000   --> Image.MAX_IMAGE_PIXELS = 95000*95000 
```


[//]: <p align="center">
[//]:   <img src="https://github.com/SNIR-NIMH/nd2totiff/blob/main/imgs/bigdata.png" height="500"/>  
[//]: </p>

## Multi-GPU Deconvolution on Cluster
On a cluster, the deconvolution can be parallelized with as many GPUs as possible over as many nodes 
as needed. With multi-GPU option, the input must be a series of 2D tifs. A single 3D tif is not accepted. 
The maximum number of GPUs is the number of slices of the image. Each slice can be deconvolved separately 
in either 2D or 3D fashion based on the size of the PSF.

This is a two-step process.
1. Use *deconvolution_cluster_prepare.py* to generate a swarm file containing N lines for N GPUs.
Each line is a call for *deconvolution_cluster_run.py* using exactly one GPU. This script should take a couple of seconds.


```
python deconvolution_cluster_prepare.py -h
usage: deconvolution_cluster_prepare.py [-h] --d INPUTDIR --psf PSF --o OUTPUTDIR [--n NUMGPU] [--iter NUMITER]
                                        [--chunks CHUNKS [CHUNKS ...]] [--float]

******************************************************************* 
Prepare for deconvolution on a cluster in parallel. Use it for 
a folder with multiple 2D images. After running this code, it will generate 
a swarm file and some json files. Run the swarm file which deconvolves 
images in parallel.
******************************************************************* 

options:
  -h, --help            show this help message and exit

Required Arguments:
  --d INPUTDIR, --img INPUTDIR
                        Input directory containing multiple 2D tif images. Note, a single 3D image is NOT acceptable.
  --psf PSF             PSF, a 3D tif image.
  --o OUTPUTDIR         Output folder where output 2D deconvolved slices will be written.

Optional Arguments:
  --n NUMGPU            Number of GPUs to run in parallel. Default 20. Usually 20-40 is alright. It depends on how many GPUs
                        are allocated per user and how big the images are.
  --iter NUMITER        Number of R-L deconvolution iterations, default 20.
  --chunks CHUNKS [CHUNKS ...]
                        Number of chunks. E.g. --chunks 4 5 indicates the image will be split into 4x5 chunks (HxW) and each
                        chunk will be deconvolved in a serial manner. This is useful for large stitched images.
  --float               If the output deconvolved images are stored as 32-bit float images instead of default 16-bit unsigned
                        integer images, use --float option.

Example:
python deconvolution_cluster_prepare.py --d /home/user/stitched/ --o /home/user/stitched_decon/ --psf psf.tif --n 20 --iter 25 --chunks 5 10
```

2. After generating a text file (a *swarm* in the context of a cluster), please submit the swarm to the cluster
according to the swarm commands. Otherwise, in a single node multi-GPU environment, edit the file to
change the **--gpu 0** argument to **--gpu 0**, **--gpu 1**, .., **--gpu N**,  etc. Then simply run
the file using GNU Parallel (or PPSS) to have all the commands run in parallel.

<!-- NOTES -->
## Notes
1. Theoretical PSF can be generated using either Fiji or Deconwolf,
```
https://bigwww.epfl.ch/algorithms/psfgenerator/
https://github.com/elgw/deconwolf
```
We have empirically found that Deconwolf provides more accurate PSF for Widefield and Confocal images.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Snehashis Roy - email@snehashis.roy@nih.gov

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REFERENCE -->
## References


