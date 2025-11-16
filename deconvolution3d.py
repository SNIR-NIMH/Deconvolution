import warnings
warnings.filterwarnings("ignore")
from time import time
import numpy as np
import os, sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
import copy
#import cupy as cp
#from cucim.skimage import img_as_float
#from cucim.skimage.restoration import richardson_lucy

from skimage.io import imsave, imread
from pytiff import Tiff
from tqdm import  tqdm
import argparse
from argparse import RawDescriptionHelpFormatter
from PIL import Image
from glob import glob
from pynvml import *

Image.MAX_IMAGE_PIXELS = 45000*45000





usage = '''Example:
1) Input and output are both 3D tif images. They are also relatively small sized, that can fit into RAM.
Normally 2560x2160x2000 images are ok with this.
python deconvolution3d.py --im A_3d_tif_stack.tif --o decon.tif --psf psf.tif   

2) Input image is in a directory, written as 2D slices. 
python deconvolution3d.py --im /home/user/a_dir_with_2D_tifs/ --o  /home/user/outputdir/ --psf psf.tif --iter 12
OR
python deconvolution3d.py --im /home/user/a_dir_with_2D_tifs/ --o  /home/user/outputimage.tif --psf psf.tif


3) Input image size is so large it will not fit into GPU memory
E.g. stitched images 8000x8000x5000 will neither fit into RAM or GPU memory, so --chunks 3 3 option could be used.
python deconvolution3d.py --im /home/user/stitched_image_dir/ --o  /home/user/outputdir/ --psf psf.tif --chunks 2 2
When chunking is used, output could be mentioned as a directory, because the output file may not fit into memory.
'''

def compute_required_memory(dim):
    dim2 = []
    for i in range(len(dim)):
        x = np.ceil(np.log2(dim[i]))
        dim2.append(2**x)
    #print('Padded matrix size = {}'.format(dim2))
    x = np.prod(np.asarray(dim2,dtype=int))*4*4/(1024**2)  # 4 for float32 bytes, algorithm needs 4x memory of input
    x = int(1.2*x)                  # 20% overhead, although it is a bad estimation
    return x



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='******************************************************************* \n'
                                     'Deconvolution using RAPIDSAI cuCIM (Compute Unified Device Architecture Clara IMage).\n'
                                     'Use this code for smaller size images, e.g. size 2560x2160x1200.\n'
                                     'For larger images (such as stitched ones), use deconvolution_cluster_prepare.py \n'
                                     'and deconvolution_cluster_run.py, which deconvolves with parallel GPUs on Biowulf cluster..\n'
                                     '******************************************************************* \n',
                                     formatter_class=RawDescriptionHelpFormatter)

    req = parser.add_argument_group('Required Arguments')
    req.add_argument('--im','-i', required=True, action='store', dest='IMAGE', type=str,
                        help='Input 3D tif image stack, or a directory containing multiple 2D tif images. ')
    req.add_argument('--psf', required=True, type=str, dest='PSF', action='store',
                        help='PSF 3D tif image.')
    req.add_argument('--o','-o', required=True, dest='OUTPUT', type=str, action='store',
                        help='Output image, either a .tif file or a folder where output 2D slices will be written. '
                             'If the output ends with .tif, it is assumed that a 3D file will be written. '
                             'Don''t use the .tif option if the image size is too big to fit in RAM, e.g. stitched images.')

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--iter', required=False, default=20, type=int, dest='NUMITER', action='store',
                        help='Number of iterations, default 10.')
    optional.add_argument('--gpu', type=int, required=False, default=0, dest='GPUID',
                        help='GPU id to use, default 0')
    optional.add_argument('--chunks', type=int, nargs='+', required = False, dest='CHUNKS', default=(1,1),
                          help='If this is mentioned, then the image is split into given chunks and deconvolved. '
                               'Use this option if the image X-Y size is too large (>2560 pixels) to fit in RAM. '
                               'If this option is used, deconvolution will be slower. ')
    optional.add_argument('--float', required=False, dest='FLOAT', action='store_true', default=False,
                        help='Use this option to save output images as FLOAT32. Default is UINT16. This is useful '
                             'if the dynamic range of the image is small. Note, saving as FLOAT32 images will '
                             'approximately double the size of the image.')

    parser.epilog=usage

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)


    results = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(results.GPUID)
    print('CUDA_VISIBLE_DEVICES set to {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    import cupy as cp  # This must come after CUDA_VISIBLE_DEVICES is set, otherwise the default gpu id 0 will be used.
    # This is not a problem in Biowulf because CUDA_VISIBLE_DEVICES is always set to 0 and only 1 GPU is allocated by swarm
    from cucim.skimage import img_as_float
    from cucim.skimage.restoration import richardson_lucy
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config)
    nvmlInit()
    gpuhandle = nvmlDeviceGetHandleByIndex(results.GPUID)
    mem = nvmlDeviceGetMemoryInfo(gpuhandle)


    if results.CHUNKS[0]>1 or results.CHUNKS[1]>1 :
        print('Image will be split into %d x %d chunks.' %(results.CHUNKS[0], results.CHUNKS[1]))
    print('Total available GPU memory = %.2f GB (%d bytes)' %(mem.total/(1024.0**3), int(mem.total)))
    if results.FLOAT == False:
        print('Output will be written in UINT16 format.')
    else:
        print('Output will be written in FLOAT32 format.')

    if os.path.isfile(results.IMAGE):
        _,ext = os.path.splitext(results.IMAGE)
        if ext == '.tif' or ext == '.tiff':
            print('Reading {}'.format(results.IMAGE))
            handle = Tiff(results.IMAGE, 'r')
            adim = (handle.number_of_pages,handle.size[0],handle.size[1])
        else:
            sys.exit('File must be a 3D .tif file')
        inputid, _ = os.path.splitext(os.path.basename(results.IMAGE))

    elif os.path.isdir(results.IMAGE):
        print('Reading {}'.format(results.IMAGE))
        files = sorted(glob(os.path.join(results.IMAGE,'*.tif')))
        # Don't use io.imread to open 2D OME tiff, use PIL. io.imread works best for 3D tiff
        # io.imread will try to open the first OME-TIFF, read all the metadata, and read all the tiffs in the folder
        x = np.asarray(Image.open(files[0]), dtype=np.uint16)
        adim = (len(files),x.shape[0], x.shape[1])
    else:
        sys.exit('ERROR: Input image ({}) can not be read.'.format(results.IMAGE))

    _,ext = os.path.splitext(results.OUTPUT)
    if ext.lower() != '.tif' and ext.lower() != '.tiff':
        os.makedirs(results.OUTPUT, exist_ok=True)
        output_is_dir = True
    else:
        output_is_dir = False

    print('Input image size = {}'.format(adim))

    kernel = imread(results.PSF,plugin='pil', is_ome=False)  # default plugin tifffile does not read ADOBEDEFLATE tif files
    kernel = np.asarray(kernel,dtype=np.float32)
    kernel = kernel/np.mean(kernel)
    kdim = kernel.shape
    kdim = np.asarray(kdim, dtype=int)

    # For a 2D PSF, force the pseudo 3D psf
    if len(kdim) ==2:
        kernel2 = np.zeros((1,kdim[0],kdim[1]), dtype=np.float32)
        kernel2[0,:,:] = kernel
        kernel = copy.deepcopy(kernel2)
        kdim = kernel.shape
        kdim = np.asarray(kdim, dtype=int)


    # This is used because sometimes io.imread can automatically reorient if xy size is smaller
    # than z size. So a 5x5x19 PSF can automatically be oriented to 19x5x5. But first channel
    # must be z, so I reorient it again.
    # This flipping is automatically done for RGB images, i.e. with 3 or 2 channel images, so needs to flip back
    # @TODO: read PSF via pytiff
    if kdim[0] > kdim[2]:
        kernel = np.transpose(kernel,(2,0,1))
        kdim = kernel.shape
    print('Original PSF image size = {}'.format(kdim))

    # Make sure the number of z slices is odd.
    if np.mod(kdim[0],2)==0:
        kernel2 = np.zeros((1+kdim[0], kdim[1], kdim[2]),dtype=np.float32)
        kernel2[0:kdim[0],:,:] = kernel
        kernel = copy.deepcopy(kernel2)
        kdim = kernel.shape
        print('Padded PSF image size = {}'.format(kdim))

    kdim2 = (kdim[0] // 2, kdim[1], kdim[2])
    #print(kdim2)



    if output_is_dir == False:
        if results.FLOAT == False:
            res = np.zeros(adim, dtype=np.uint16)
        else:
            res = np.zeros(adim, dtype=np.float32)

    if output_is_dir == True:
        mem = 4 * (2 * kdim2[0] + 2) * (adim[1] + 2 * kdim[1] + 1) * (adim[2] + 2 * kdim[2] + 1)  # max memory required is slightly more than the float32 memory
        mem = int(np.ceil(1.2 * mem / (1024 ** 3)))
        mem = np.ceil(mem / 10) * 10
        print('Maximum memory required is %d GB' % (mem))
    else:
        if results.FLOAT == False:
            mem = 8* 4 * (2 * kdim2[0] + 2) * (adim[1] + 2 * kdim[1] + 1) * (adim[2] + 2 * kdim[2] + 1) + 2 * np.prod(adim)
            # 8 float32 temp variables with (kdim[0],adim[1],adim[2])
        else:

            mem = 8 * 4 * (2 * kdim2[0] + 2) * (adim[1] + 2 * kdim[1] + 1) * (adim[2] + 2 * kdim[2] + 1) \
                  + 4 * np.prod(adim)
        mem = int(np.ceil(1.2 * mem / (1024 ** 3)))
        mem = np.ceil(mem / 10) * 10
        print('Maximum memory required is %d GB' % (mem))


    print('Initializing..')
    #algo = fd_restoration.RichardsonLucyDeconvolver(3, pad_mode='none').initialize()



    numsplit = np.asarray(results.CHUNKS,dtype=int)
    sdim = (numsplit[0] * numsplit[1], adim[0], adim[1] // numsplit[0] + 2 * kdim[1] + 1,
                    adim[2] // numsplit[1] + 2 * kdim[2] + 1)
    print('Number of splits = %d x %d ' % (numsplit[0], numsplit[1]))
    print('Split input slices from {} to {}'.format(adim, sdim))
    print('Temp variable size = {} x {} x {}'.format(2*kdim2[0]+1, adim[1], adim[2]))
    print('Approximate GPU memory required = {} MB'.format(compute_required_memory((2 * kdim[0] + 1, sdim[2], sdim[3]))))


    print('Prefetching..')
    t1 = np.zeros((2 * kdim2[0] + 1, adim[1], adim[2]), dtype=np.float32)  # previous variable
    # Initialize t1 by filling half of it starting from kdim2[0]-th element, i.e. kdim2[0]+1
    #@TODO : Repeat  boundary condition by copying the first frame, not zeroes
    if os.path.isfile(results.IMAGE):
        for j in tqdm(range(0, kdim2[0]+1)):

            handle.set_page(j)
            t1[j + kdim2[0], :, :] = handle[:]
    else:
        for j in tqdm(range(0, kdim2[0]+1)):
            x = np.asarray(imread(files[j], is_ome=False, plugin='pil'), dtype=np.uint16)  # is_ome=False is mandatory here
            #x = np.asarray(Image.open(files[j]), dtype=np.uint16)  # Don't use io.imread to open OME-TIFF files
            t1[j + kdim2[0], :, :] = np.asarray(x, dtype=np.float32)
    for j in range(0,kdim2[0]):  # Pad by 1st slice
        t1[j,:,:] = t1[kdim2[0],:,:]

    print('Running..')
    T0 = time()


    #y = cp.asarray(img_as_float(np.asarray(imread('PSF_BatSPIM2.tif'), dtype=np.float32)))
    cukernel=cp.asarray(img_as_float(np.asarray(kernel, dtype=np.float32)))

    for k in tqdm(range(0,adim[0])):
        if results.FLOAT == True:
            out = np.zeros((adim[1], adim[2]), dtype=np.float32)
        else:
            out = np.zeros((adim[1], adim[2]), dtype=np.uint16)

        if results.CHUNKS[0]>1 or results.CHUNKS[1]>1:

            sdim = (2*kdim2[0]+1, adim[1] // numsplit[0] + 2 * kdim[1] + 1, adim[2] // numsplit[1] + 2 * kdim[2] + 1)

            splitinput = np.zeros(sdim, dtype=np.float32)

            count = 0
            for i in range(0, numsplit[0]):
                for j in range(0, numsplit[1]):
                    I1 = i * (adim[1] // numsplit[0]) - kdim[1]
                    I2 = (i + 1) * (adim[1] // numsplit[0]) + kdim[1] + 1
                    J1 = j * (adim[2] // numsplit[1]) - kdim[2]
                    J2 = (j + 1) * (adim[2] // numsplit[1]) + kdim[2] + 1
                    delta = [0, 0]
                    if I1 < 0:
                        delta[0] = -I1
                        I1 = 0
                    if I2 > adim[1]:
                        I2 = adim[1]
                    if J1 < 0:
                        delta[1] = -J1
                        J1 = 0
                    if J2 > adim[2]:
                        J2 = adim[2]
                    x = t1[:, I1:I2, J1:J2]
                    splitinput[:, delta[0]:x.shape[1] + delta[0], delta[1]:x.shape[2] + delta[1]] = x

                    cusplitinput = cp.asarray(img_as_float(np.asarray(splitinput, dtype=np.float32)))
                    x = richardson_lucy(cusplitinput, cukernel, num_iter=results.NUMITER, clip=False, filter_epsilon=1e-6)
                    x = cp.asnumpy(x)

                    #x = algo.run(fd_data.Acquisition(data=splitinput, kernel=kernel), niter=results.NUMITER, session_config=config).data
                    #x = np.asarray(x, dtype=np.uint16)
                    I1 = i * (adim[1] // numsplit[0])
                    I2 = (i + 1) * (adim[1] // numsplit[0])
                    J1 = j * (adim[2] // numsplit[1])
                    J2 = (j + 1) * (adim[2] // numsplit[1])
                    #out[I1:I2, J1:J2] = x[kdim[0], kdim[1]:-kdim[1] - 1, kdim[2]:-kdim[2] - 1]
                    temp = x[kdim2[0], kdim[1]:-kdim[1] - 1, kdim[2]:-kdim[2] - 1]
                    if results.FLOAT == True:
                        out[I1:I2, J1:J2] = temp
                    else:
                        temp[temp > 65535] = 65535
                        temp[temp < 0] = 0
                        out[I1:I2, J1:J2] = np.asarray(temp, dtype=np.uint16)

                    if count==0 and k==0: # Print only once
                        mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                        print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)),
                                                                  int(mem.free / (1024 ** 2))))
                    count = count + 1


        else:
            cut1 = cp.asarray(img_as_float(np.asarray(t1, dtype=np.float32)))
            x = richardson_lucy(cut1, cukernel, num_iter=results.NUMITER, clip=False, filter_epsilon=1e-6)
            x = cp.asnumpy(x)
            #x = algo.run(fd_data.Acquisition(data=t1, kernel=kernel), niter=results.NUMITER,
            #                session_config=config).data
            out = x[kdim2[0],:,:]
            if k==0:
                mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)), int(mem.free / (1024 ** 2))))

        if output_is_dir == False:  # 3D image
            if results.FLOAT == False:
                out[out<0]=0
                out[out>65535] = 65535
                res[k, :, :] = np.asarray(out, dtype=np.uint16)
            else:
                res[k, :, :] = out
        else:  # 2D slices
            if os.path.isfile(results.IMAGE) == True:
                s = inputid + '_decon_Z' + str(k).zfill(4) + '.tif'
                s = os.path.join(results.OUTPUT, s)
            else:
                s = os.path.basename(files[k])
                s,_ = os.path.splitext(s)
                s = s + '_decon.tif'
                s = os.path.join(results.OUTPUT,s)

            if results.FLOAT == False:
                out[out<0]=0
                out[out>65535] = 65535
                out = np.asarray(out, dtype=np.uint16)
            if results.FLOAT == False:
                imsave(s, out, check_contrast=False, compression='zlib')
            else: # for 32bit images, don't compress
                imsave(s, out, check_contrast=False)

        # Update t1, only change the last image
        for j in range(0, 2 * kdim2[0]):
            t1[j, :, :] = t1[j + 1, :, :]  # update current variable with previous variable, only read the last image, shift previous ones

        if os.path.isfile(results.IMAGE):
            idx = k + kdim2[0] + 1
            # print(idx)
            if idx >= 0 and idx < adim[0]:  # This means the boundary at the end is repeated.
                handle.set_page(idx)
                t1[2 * kdim2[0], :, :] = handle[:]
        else:
            idx = k + kdim2[0] + 1
            # print(idx)
            if idx >= 0 and idx < adim[0]:
                x = imread(files[idx], is_ome=False, plugin='pil')
                t1[2 * kdim2[0], :, :] = np.asarray(x, dtype=np.float32)

    T1 = time()
    print('Deconvolution time = %.2f seconds ' % (T1 - T0))

    if output_is_dir == False:
        print('Writing {}'.format(results.OUTPUT))
        T0 = time()

        if results.FLOAT == False:
            if 2 * np.prod(adim) >= 4 * (1024 ** 3):
                print('Writing bigtif format, image size could be > 4GB ')
                imsave(results.OUTPUT, res, check_contrast=False, compression='zlib', bigtiff=True)
            else:
                imsave(results.OUTPUT, res, check_contrast=False, compression='zlib', bigtiff=False)
        else:  # For floating point images, compression isn't allowed
            if 4 * np.prod(adim) >= 4 * (1024 ** 3):  # float32 = 4 * size of image
                print('Writing bigtif format, image size could be > 4GB ')
                imsave(results.OUTPUT, res, check_contrast=False, bigtiff=True)
            else:
                imsave(results.OUTPUT, res, check_contrast=False, bigtiff=False)

        T1 = time()
        print('File write time = %.2f seconds ' % (T1 - T0))



