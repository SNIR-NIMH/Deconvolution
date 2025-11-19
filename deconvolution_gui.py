import os
from glob import  glob
import sys
from PIL import Image
from tqdm import tqdm
import argparse
from skimage.io import imread, imsave
import numpy as np
#from pytiff import Tiff
#import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
x = tf.test.is_gpu_available()
#x = tf.config.list_logical_devices()
if x == False:
    #print('WARNING: GPU is not available')
    sys.exit('ERROR: GPU is not available. Deconvolution will not work')

warnings.filterwarnings("ignore")

path = os.path.dirname(sys.argv[0])
path = os.path.abspath(path)
#print('Appending {}'.format(path))
sys.path.append(path)



# ======================================================================
root = tk.Tk()
root.title('GPU based 2D/3D Deconvolution')
# setting the windows size
root.geometry("1000x200")  # Width x height


def getInputFolderPath():
    folder_selected = filedialog.askdirectory()
    inputpath.set(folder_selected)

def getOutputFolderPath():
    folder_selected = filedialog.askdirectory()
    outputpath.set(folder_selected)

def getPSFPath():
    folder_selected = filedialog.askopenfilename(filetypes=[('Tif files','*.tif'),('Tif files','*.tiff')])
    psfpath.set(folder_selected)


def submit():
    inputdir = inputpath.get()
    outputdir = outputpath.get()
    psf = psfpath.get()
    g = gpuid.get()

    n = niter.get()

    c = chunks.get()
    path = os.path.dirname(sys.argv[0])
    path = os.path.abspath(path)
    path = os.path.join(path, 'deconvolution3d.py ')
    # Use --atlasdir="path" --> The double quote and equal-to ensures the space in the path is respected
    # Using --atlasdir path or --atlasdir "path" does not work if there are spaces in  path, only arg equalto quote path unquote works
    cmd = 'python ' + path + ' --im="' + inputdir + '" --o="' + outputdir + '" --psf="' + psf + '" --iter ' + str(n) +\
          ' --gpu ' + g + ' --chunks ' + str(c) + ' ' + str(c)
    print(cmd)
    os.system(cmd)

    root.destroy()


if __name__ == "__main__":


    # declaring string variable
    # for storing name and password
    inputpath = tk.StringVar()
    outputpath = tk.StringVar()
    psfpath = tk.StringVar()
    niter = tk.IntVar()
    chunks = tk.IntVar()
    gpuid = tk.StringVar()



    # creating a label for name using widget Label
    #input_label = tk.Label(root, text='Input folder')
    # creating a entry for input name using widget Entry
    #input_entry = tk.Entry(root,textvariable=input_var)

    a = tk.Label(root, text="Input image folder containing 2D slices", padx=30)
    a.grid(row=1, column=1)
    E = tk.Entry(root, textvariable=inputpath, width=50)
    E.grid(row=1, column=2, ipadx=60)
    btnFind = ttk.Button(root, text="Browse Folder", command=getInputFolderPath)
    btnFind.grid(row=1, column=3)

    a = tk.Label(root, text="Output folder where deconvolved slices will be written", padx=30)
    a.grid(row=2, column=1)
    E = tk.Entry(root, textvariable=outputpath,  width=50)
    E.grid(row=2, column=2, ipadx=60)
    btnFind = ttk.Button(root, text="Browse Folder", command=getOutputFolderPath)
    btnFind.grid(row=2, column=3)

    a = tk.Label(root, text="PSF (a 2D/3D tif for 2D/3D decon)", padx=30)
    a.grid(row=3, column=1)
    E = tk.Entry(root, textvariable=psfpath,  width=50)
    E.grid(row=3, column=2, ipadx=60)
    btnFind = ttk.Button(root, text="Browse File", command=getPSFPath)
    btnFind.grid(row=3, column=3)

    niter_label = tk.Label(root, text='Number of iterations')
    chunks_label = tk.Label(root, text='Number of chunks in either x or y direction')
    gpuid_label = tk.Label(root, text='GPU ID to use (starting from 0)')


    niter_entry = tk.Entry(root, textvariable=niter, width=10)
    chunks_entry = tk.Entry(root, textvariable=chunks, width=10)
    gpuid_entry = tk.Entry(root, textvariable=gpuid, width=10)

    niter.set(12)
    chunks.set(2)
    gpuid.set(0)


    #c = ttk.Button(root, text="find", command=doStuff)
    #c.grid(row=1, column=4)


    # creating a button using the widget
    # Button that will call the submit function
    sub_btn = tk.Button(root, text='Run', command=submit)


    niter_label.grid(row=4,column=1, padx=60)
    chunks_label.grid(row=5, column=1, padx=60)
    gpuid_label.grid(row=6, column=1, padx=60)

    niter_entry.grid(row=4, column=2, padx=60)
    chunks_entry.grid(row=5, column=2, padx=60)
    gpuid_entry.grid(row=6, column=2, padx=60)


    sub_btn.grid(row=7,column=2)

    # performing an infinite loop
    # for the window to display
    root.mainloop()
