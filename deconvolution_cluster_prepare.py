import numpy as np
from glob import glob
import argparse
import os, sys
from skimage import io
import json
from PIL import Image
import tempfile
import copy
import time
Image.MAX_IMAGE_PIXELS = 99000*99000
# Fiji has max pixel size as 2GB = 2*(1024^3)
# So any image should be < sqrt(2*1024*1024*1024) = 46340

fd_decon_path = os.path.dirname(sys.argv[0])
fd_decon_path = os.path.abspath(os.path.realpath(os.path.expanduser(fd_decon_path)))
fd_decon = os.path.join(fd_decon_path, 'deconvolution_cluster_run.py')
if os.path.isfile(fd_decon) == False:
    sys.exit('ERROR: deconvolution_cluster_run.py not found in {}'.format(fd_decon_path))

usage='''
Example:
python deconvolution_cluster_prepare.py --d /home/user/stitched/ --o /home/user/stitched_decon/ --psf psf.tif --n 20 --iter 25 --chunks 2 3
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='******************************************************************* \n'
                                     'Prepare for deconvolution on a cluster in parallel. Use it for \n'
                                                 'a folder with multiple 2D images. After running this code, it will generate \n'
                                                 'a swarm file and some json files. Run the swarm file which deconvolves \n'
                                                 'images in parallel.\n'
                                                 '******************************************************************* \n',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    req = parser.add_argument_group('Required Arguments')
    req.add_argument('--d','--img', required=True, action='store', dest='INPUTDIR', type=str,
                     help='Input directory containing multiple 2D tif images. Note, a single 3D image is NOT acceptable.')
    req.add_argument('--psf', required=True, type=str, dest='PSF', action='store',
                     help='PSF, a 3D tif image.')
    req.add_argument('--o', required=True, dest='OUTPUTDIR', type=str, action='store',
                     help='Output folder where output 2D deconvolved slices will be written.')
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--n', required = False, default=20, dest='NUMGPU', type=int,
                     help='Number of GPUs to run in parallel. Default 20. Usually 20-40 is alright. '
                          'It depends on how many GPUs are allocated per user and how big the images are.')
    optional.add_argument('--iter', required=False, default=20, type=int, dest='NUMITER', action='store',
                          help='Number of R-L deconvolution iterations, default 20.')
    optional.add_argument('--chunks', type=int, required=False, dest='CHUNKS', default=(2,2), nargs='+',
                          help='Number of chunks. E.g. --chunks 4 5 indicates the image will be split into 4x5 chunks (HxW) '
                               'and each chunk will be deconvolved in a serial manner. This is useful for large stitched '
                               'images.')
    optional.add_argument('--float', action='store_true', required=False, default=False, dest='FLOAT',
                          help='If the output deconvolved images are stored as 32-bit float images instead of '
                               'default 16-bit unsigned integer images, use --float option.')
    parser.epilog=usage

    results = parser.parse_args()

    uid = time.strftime('%d-%m-%Y_%H-%M-%S')

    if os.path.isdir(results.OUTPUTDIR) == False:
        os.makedirs(results.OUTPUTDIR)

    if os.path.isdir(results.INPUTDIR) == False:
        sys.exit('Input must be a directory containing 2D tif images.')

    results.INPUTDIR = os.path.abspath(os.path.realpath(os.path.expanduser(results.INPUTDIR)))
    results.OUTPUTDIR = os.path.abspath(os.path.realpath(os.path.expanduser(results.OUTPUTDIR)))
    results.PSF = os.path.abspath(os.path.realpath(os.path.expanduser(results.PSF)))

    kernel = np.asarray(io.imread(results.PSF), dtype=np.uint16)
    kdim = kernel.shape
    # For a 2D PSF, force the pseudo 3D psf
    if len(kdim) == 2:
        kernel2 = np.zeros((1, kdim[0], kdim[1]), dtype=np.float32)
        kernel2[0, :, :] = kernel
        kernel = copy.deepcopy(kernel2)
        kdim = kernel.shape
        kdim = np.asarray(kdim, dtype=int)

    if kdim[0] > kdim[2]:
        kernel = np.transpose(kernel, (2, 0, 1))
        kdim = kernel.shape
    print('PSF dimension (DxHxW) = {}'.format(kdim))
    # Make sure the number of z slices is odd.
    if np.mod(kdim[0], 2) == 0:
        kernel2 = np.zeros((1 + kdim[0], kdim[1], kdim[2]), dtype=np.float32)
        kernel2[0:kdim[0], :, :] = kernel
        kernel = copy.deepcopy(kernel2)
        kdim = kernel.shape
        print('Padded PSF image size = {}'.format(kdim))

    kdim2 = (kdim[0] // 2, kdim[1], kdim[2])
    print(kdim2)


    files = sorted(glob(os.path.join(results.INPUTDIR,'*.tif')))
    if len(files) == 0:
        files = sorted(glob(os.path.join(results.INPUTDIR, '*.tiff')))

    if len(files) == 0:
        sys.exit('ERROR: Input folder does not contain any .tif or .tiff files')

    try:
        x = np.asarray(Image.open(files[-1]), dtype=np.uint16)
    except:
        try:
            x = np.asarray(io.imread(files[-1]), dtype=np.uint16)
            # io.imread occasionally can't read very big Matlab created zipped tif file
            # or OME tiff images. Therefore try pillow.Image first.
            # but PIL Image can not read images bigger than 46340x46340
            # try io.imread for those cases.
        except Exception as e:
            sys.exit('ERROR: Pillow and skimage.io can not read image.'.format(files[-1]))
    if len(x.shape) != 2:
        sys.exit('Input directory must contain 2D images.')

    print('Input images dimension (DxHxW) %d x %d x %d' % (len(files),x.shape[0],x.shape[1]))

    mem = 4 * (2*kdim[0]+2)*(x.shape[0]+2*kdim[1]+1)*(x.shape[1]+2*kdim[2]+1)  # max memory required is slightly more than the float32 memory
    mem = int(np.ceil(1.2*mem / (1024**3)))
    mem = np.ceil(mem/10)*10
    print('Maximum memory required is %d GB' %(mem))

    n = len(files)
    K = 2*kdim2[0] + 1  # K = kdim[0]
    filelist = [[None]*K]*n

    if results.NUMGPU >=n:
        print('WARNING: Number of images (%d) is less than number of parallel processes (%d). Are you sure?' %(n, results.NUMGPU))
        results.NUMGPU = n

    for i in range(0,n):
        temp = [None]*K
        count=0
        for j in range(i-kdim2[0],i+kdim2[0]+1):
            if j>=0 and j<n:
                temp[count]  = files[j]
            elif j<0:
                temp[count] = files[0]
            elif j>=n:
                temp[count] =files[n-1]
            count = count + 1
        filelist[i]=temp


    delta = int(np.ceil(n/results.NUMGPU))
    count=0
    jsons=[]
    for i in range(0,n,delta):
        j = i+delta
        if j>=n:
            j=n
        temp = filelist[i:j]
        s = 'filenames_' + uid + '_' + str(count+1).zfill(3) + '.json'
        s = os.path.join(results.OUTPUTDIR,s)
        jsons.append(s)
        print('Writing {}'.format(s))
        with open(s,'w') as f:
            json.dump(temp,f)
        count = count + 1

    s = 'swarm_' + uid + '.swarm'
    s = os.path.join(results.OUTPUTDIR,s)
    f1 = open(s,'w+')
    for i in range(0,count):
        if results.FLOAT ==True:
            print('python %s --js %s --o %s --psf %s --iter %d --gpu 0 --chunks %d %d --float'
              %(fd_decon,jsons[i], results.OUTPUTDIR, results.PSF, results.NUMITER, results.CHUNKS[0], results.CHUNKS[1] ), file=f1)
        else:
            print('python %s --js %s --o %s --psf %s --iter %d --gpu 0 --chunks %d %d '
                  % (fd_decon, jsons[i], results.OUTPUTDIR, results.PSF, results.NUMITER, results.CHUNKS[0],
                     results.CHUNKS[1]), file=f1)
        #print('python %s --js %s --o %s --psf %s --iter %d --gpu 0 --chunks %d %d'
        #      %(fd_decon,jsons[i], results.OUTPUTDIR, results.PSF, results.NUMITER, results.CHUNKS[0], results.CHUNKS[1] ), file=f1)

    f1.close()
    print(':============================')
    print('Now run this on a login node shell:')
    print('swarm -f %s --partition=gpu --gres=gpu:k80:1  --merge-output -t 4 -g %d  --time 8:00:00' %(s, mem))
    print(':============================')