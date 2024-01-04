#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:45:34 2023

@author: ubuntu
"""

import os
import glob

#input_folder = '../Mustafa_segmentation/*reviewed';
#input_folder = '../Udit_Segmentation/*reviewed';



input_folder = '../Data/*';

import nrrd
import re

image_list = glob.glob(input_folder)
dest_label_folder = '../Dataset/labelsTr/'
dest_image_folder = '../Dataset/imagesTr/'

os.makedirs(dest_label_folder, exist_ok=True)
os.makedirs(dest_image_folder, exist_ok=True)


import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np
for fname in sorted(image_list):
    #print(os.path.dirname(fname))
    #print(os.path.basename(fname))
    subject_name = os.path.basename(fname)
    subject_name = subject_name.replace(' ', '_')
    print(subject_name, fname)
    
    print(len(glob.glob(fname + '/*.nii.gz') + glob.glob(fname + '/*.nii') )) 
    img_data_list = glob.glob(fname + '/*.nii.gz') + glob.glob(fname + '/*.nii') 
    img_data_fn = img_data_list[0]
    label_fn = subject_name
    nii_orig = nib.load(img_data_fn)
    #final data file
    #nib.save(nii_orig, dest_image_folder + label_fn+'_0000.nii.gz')
    
    
    
    label_data_list = glob.glob(fname + '/*.nrrd')
    label_data_fn = label_data_list[0]
    
    

    loadedSegmentationNode = nrrd.read(label_data_fn)
    nrrd_data = loadedSegmentationNode[0]
    header = loadedSegmentationNode[1]
    cube_size = header["sizes"]

    image_offset = header["Segmentation_ReferenceImageExtentOffset"]
    result = re.finditer(r'[\s]', image_offset)
    
    '''
    offset_list = []
    xxx = 0
    for m in result:
        print(m.start(0))
        #print(m.)
        offset_list.append(int(image_offset[xxx:m.start(0)]))
        xxx = m.start(0)+1
        offset_list.append(int(image_offset[xxx:]))
    
    print(header["sizes"])
    print(header["Segmentation_ReferenceImageExtentOffset"])
    '''
    #
    
    #print('original data size ', nii_orig.shape)
    print('original data size ', nii_orig.shape, ' segment size ', header["sizes"], ' Segmentation_ReferenceImageExtentOffset ', header["Segmentation_ReferenceImageExtentOffset"])

    #nii_orig = nib.load(fname_final[0])
    loadedVolumeNode  = nii_orig
    #print(loadedVolumeNode.affine)
    my_data = loadedVolumeNode.get_fdata() -loadedVolumeNode.get_fdata()
    #my_data[offset_list[0]:offset_list[0]+header["sizes"][0],offset_list[1]:offset_list[1]+header["sizes"][1], offset_list[2]:offset_list[2] + header["sizes"][2]] = nrrd_data        
    my_data = nrrd_data
    seg_resized_nii = nib.Nifti1Image(my_data, loadedVolumeNode.affine, loadedVolumeNode.header)
    seg_path = dest_label_folder + label_fn+'.nii.gz'
    nib.save(seg_resized_nii, seg_path)

    '''
    label_fn = subject_name #[:-9]    
    #==================================Converting label 
    fname_final = fname + '/Segmentation.revised_seg.nii.gz'    
    #print(fname_final)
    nii_orig = nib.load(fname_final)
    
    nib.save(nii_orig, dest_label_folder + label_fn+'.nii.gz')
    print(dest_label_folder + label_fn+'.nii.gz') 
    #==================================Converting image from nii to nii.gz  
    fname_final = glob.glob(fname + '/*.nii')
    
    if(len(fname_final)==0):
        fname_final = glob.glob(fname + '/*.nii.gz')
    
#    fname_final = glob.glob(fname + '/segmentation/*.nii')
    #print(fname_final)
    #print(fname)
    nii_orig = nib.load(fname_final[0])
    #data = _nrrd[0]
    #header = _nrrd[1]
    
    #img = nib.Nifti1Image(data, np.eye(4))
    #label_fn = subject_name[:-9]
    #print(label_fn)
    
    #print(dest_image_folder + label_fn+'.nii.gz')
    #nib.save(nii_orig, dest_image_folder + label_fn+'.nii.gz')
    print(dest_image_folder + label_fn+'.nii.gz')
    
    '''    
print('total subjects ', len(image_list))