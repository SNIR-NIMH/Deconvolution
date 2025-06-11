import json
import warnings
warnings.filterwarnings("ignore")
from time import time
import numpy as np
import os, sys
#print("Python version")
#print (sys.version)
#print("Version info.")
#print (sys.version_info)
#os.system('which python')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
import copy
import cupy as cp
from cucim.skimage import img_as_float
from cucim.skimage.restoration import richardson_lucy

from skimage.io import imsave, imread
from pytiff import Tiff
from tqdm import  tqdm
import argparse
from argparse import RawDescriptionHelpFormatter
from PIL import Image
from glob import glob
from pynvml import *

Image.MAX_IMAGE_PIXELS = 99000*99000
#Image.MAX_IMAGE_PIXELS = 46340*46340
# Fiji has max pixel size as 2GB = 2*(1024^3)
# So any image should be < sqrt(2*1024*1024*1024) = 46340




usage = '''Example:

Input image size is so large it will not fit into GPU memory:

E.g. stitched images 8000x8000x5000 will neither fit into RAM or GPU memory, so --chunks dH dW option is needed.
Then the images will be split into dH x dW (height x width) overlapping chunks and each chunk will be individually serially deconvolved.
python deconvolution_cluster_prepare.py --im /home/user/stitched_image_dir/ --o  /home/user/outputdir/ --psf psf.tif --chunks 10 10


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
                                     'Deconvolution using RAPIDSAI cuCIM (Compute Unified Device Architecture Clara IMage) for very large images. \n'
                                     'See deconvolution_cluster_prepare.py for more info.\n'
                                     '******************************************************************* \n',
                                     formatter_class=RawDescriptionHelpFormatter)

    req = parser.add_argument_group('Required Arguments')
    req.add_argument('--js', required=True, action='store', dest='IMAGE', type=str,
                        help='Input json file containing filenames to deconvolve. '
                             'This is usually created by deconvolution_cluster_prepare.py. ')
    req.add_argument('--psf', required=True, type=str, dest='PSF', action='store',
                        help='PSF 3D tif image.')
    req.add_argument('--o', required=True, dest='OUTPUTDIR', type=str, action='store',
                        help='Output folder where output 2D slices will be written. '
                             'This must be the same folder that was used in deconvolution_cluster_prepare.py')

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--iter', required=False, default=20, type=int, dest='NUMITER', action='store',
                        help='Number of iterations, default 20.')
    optional.add_argument('--gpu', type=int, required=False, default=0, dest='GPUID',
                        help='GPU id to use, default 0')
    optional.add_argument('--chunks', type=int, required = False, dest='CHUNKS', default=(1,1),nargs='+',
                          help='Number of chunks. E.g. --chunks 4 5 indicates the image will be split into 4x5 chunks (HxW) '
                               'and each chunk will be deconvolved in a serial manner. This is useful for large stitched '
                               'images.')
    optional.add_argument('--float', action='store_true', required=False, default=False, dest='FLOAT',
                          help='If the output deconvolved images are stored as 32-bit float images instead of '
                               'default 16-bit unsigned integer images, use --float option.')
    parser.epilog=usage

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)


    results = parser.parse_args()



    os.environ['CUDA_VISIBLE_DEVICES'] = str(results.GPUID)
    print('CUDA_VISIBLE_DEVICES set to {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config)
    nvmlInit()
    gpuhandle = nvmlDeviceGetHandleByIndex(results.GPUID)
    mem = nvmlDeviceGetMemoryInfo(gpuhandle)


    if results.CHUNKS[0]>1 or results.CHUNKS[1]>1 :
        print('Image will be split into chunks.')
    print('Total available GPU memory = %.2f GB (%d bytes)' %(mem.total/(1024.0**3), int(mem.total)))


    if os.path.isfile(results.IMAGE):
        _,ext = os.path.splitext(results.IMAGE)
        if ext == '.json':
            print('Reading {}'.format(results.IMAGE))
            f1 = open(results.IMAGE)
            filelist = json.load(f1)
            f1.close()
            for u in range(0,len(filelist[0])):
                if filelist[0][u] != None:
                    try:
                        x = np.asarray(Image.open(filelist[0][u]), dtype=np.uint16)
                    except:
                        try:
                            x = np.asarray(imread(filelist[0][u], is_ome=False), dtype=np.uint16)
                        except Exception as e:
                            print('ERROR: Pillow and skimage.io can not read image.'.format(filelist[0][u]))
                            sys.exit(str(e))
                    x = np.asarray(x, dtype=np.uint16)

                    break
            adim = (len(filelist),x.shape[0],x.shape[1])

        else:
            sys.exit('File must be a 3D .tif file')

    else:
        sys.exit('ERROR: Input json file ({}) can not be read.'.format(results.IMAGE))


    if os.path.isdir(results.OUTPUTDIR) == False:
        os.makedirs(results.OUTPUTDIR, exist_ok=True)


    print('Input image size = {}'.format(adim))

    kernel = imread(results.PSF)
    kernel = np.asarray(kernel,dtype=np.float32)
    kdim = np.asarray(kernel.shape, dtype=int)
    # For a 2D PSF, force the pseudo 3D psf
    if len(kdim) == 2:
        kernel2 = np.zeros((1, kdim[0], kdim[1]), dtype=np.float32)
        kernel2[0, :, :] = kernel
        kernel = kernel2
        kdim = kernel.shape
        kdim = np.asarray(kdim, dtype=int)

    if kdim[0] > kdim[2]:
        kernel = np.transpose(kernel,(2,0,1))
        kdim = kernel.shape
    print('PSF image size = {}'.format(kdim))

    # Make sure the number of z slices is odd.
    if np.mod(kdim[0], 2) == 0:
        kernel2 = np.zeros((1 + kdim[0], kdim[1], kdim[2]), dtype=np.float32)
        kernel2[0:kdim[0], :, :] = kernel
        kernel = copy.deepcopy(kernel2)
        kdim = kernel.shape
        print('Padded PSF image size = {}'.format(kdim))

    kdim2 = (kdim[0] // 2, kdim[1], kdim[2])
    print(kdim2)


    if len(filelist[0]) != 2*kdim2[0]+1:
        sys.exit('ERROR: Final PSF dimension (%d) must be same as length of files in the json file (%d)'
                 %(kdim[0], len(filelist)))

    numsplit = np.asarray(results.CHUNKS, dtype=int)
    if numsplit[0] < 1:
        numsplit[0] = 1
    if numsplit[1] < 1:
        numsplit[1] = 1

    #print('Initializing..')
    #algo = fd_restoration.RichardsonLucyDeconvolver(3, pad_mode='none').initialize()



    if numsplit[0]<1:
        numsplit[0] = 1
    if numsplit[1]<1:
        numsplit[1]=1

    sdim = (numsplit[0] * numsplit[1], adim[0], adim[1] // numsplit[0] + 2 * kdim[1] + 1,
            adim[2] // numsplit[1] + 2 * kdim[2] + 1)
    print('Number of splits = %d x %d ' % (numsplit[0], numsplit[1]))
    print('Split input slices from {} to {}'.format(adim, sdim))
    print('Approximate GPU memory required = {} MB'.format(compute_required_memory((2 * kdim2[0] + 1, sdim[2], sdim[3]))))

    cukernel = cp.asarray(img_as_float(np.asarray(kernel, dtype=np.float32)))

    print('Running..')
    T0 = time()
    for k in tqdm(range(0,adim[0])):
        # Redefine output matrix to zero for every slice, because the numbers should not accumulate
        if results.FLOAT == True:
            out = np.zeros((adim[1], adim[2]), dtype=np.float32)
        else:
            out = np.zeros((adim[1], adim[2]), dtype=np.uint16)
        t1 = np.zeros((2 * kdim2[0] + 1, adim[1], adim[2]), dtype=np.float32)  # initialize to zero

        for p in range(0,2*kdim2[0]+1):
            if filelist[k][p] != None:
                for u in range(0, len(filelist[0])):
                    if filelist[0][u] != None:
                        try:
                            #x = np.asarray(imread(filelist[k][p], is_ome=False, plugin='pil'), dtype=np.uint16)
                            x = np.asarray(imread(filelist[k][p], is_ome=False), dtype=np.uint16) # pillow is very slow
                        except:
                            try:
                                x = np.asarray(Image.open(filelist[k][p]), dtype=np.uint16)
                            except Exception as e:
                                print('ERROR: Pillow and skimage.io can not read image {}'.format(filelist[k][p]))
                                sys.exit(str(e))
                #x = np.asarray(Image.open(filelist[k][p]), dtype=np.uint16)
                t1[p,:,:] = np.asarray(x, dtype=np.float32)

        if results.CHUNKS[0]>1 or results.CHUNKS[1]>1:

            sdim = (2*kdim2[0]+1, adim[1] // numsplit[0] + 2 * kdim[1] + 1, adim[2] // numsplit[1] + 2 * kdim[2] + 1)



            count = 0
            for i in range(0, numsplit[0]):
                for j in range(0, numsplit[1]):
                    splitinput = np.zeros(sdim, dtype=np.float32)
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
                    x = richardson_lucy(cusplitinput, cukernel, num_iter=results.NUMITER, clip=False,
                                        filter_epsilon=1e-6)
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
            #x = algo.run(fd_data.Acquisition(data=t1, kernel=kernel), niter=results.NUMITER,
            #                session_config=config).data
            cut1 = cp.asarray(img_as_float(np.asarray(t1, dtype=np.float32)))
            x = richardson_lucy(cut1, cukernel, num_iter=results.NUMITER, clip=False, filter_epsilon=1e-6)
            x = cp.asnumpy(x)

            out = x[kdim2[0],:,:]
            if results.FLOAT == False:
                out[out < 0] = 0
                out[out > 65535] = 65535
                out = np.asarray(out, dtype=np.uint16)
            if k==0:
                mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)), int(mem.free / (1024 ** 2))))


        s = os.path.basename(filelist[k][kdim2[0]])
        s,_ = os.path.splitext(s)
        s = s + '_decon.tif'
        s = os.path.join(results.OUTPUTDIR,s)

        if 2 * adim[1]*adim[2] >= 4 * (1024 ** 3):
            imsave(s, out, check_contrast=False, compression='zlib', bigtiff=True)
        else:
            imsave(s, out, check_contrast=False, compression='zlib', bigtiff=False)




    T1 = time()
    print('Deconvolution time = %.2f seconds ' %(T1-T0))
