#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:22:16 2023

@author: zewang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:16:52 2022
@author: yiran
"""
#import slicer
import nibabel as nib
import os
import glob
import re
import nrrd
import numpy as np
count = 0
import SimpleITK as sitk

#This implementation is based on sitk 
def nrrd2nii_itk(nrrd_fn, dest_folder, final_name):
    #loadedSegmentationNode = nrrd.read(nrrd_fn)
    img = sitk.ReadImage(nrrd_fn)
    os.makedirs(dest_folder, exist_ok=True)
    print(dest_folder + final_name)
    sitk.WriteImage(img, dest_folder + final_name)
    
    
    



#This is low level implementation
def nrrd2nii(ref_fn, nrrd_fn, dest_folder , prefix=''):
    #gt_fn_list = glob.glob(fname+'/*Nathan Segmentation.seg*.nrrd')    
    #print('', gt_fn_list[0])    
    loadedSegmentationNode = nrrd.read(nrrd_fn)
    nrrd_data = loadedSegmentationNode[0]
    header = loadedSegmentationNode[1]
    cube_size = header["sizes"]
    image_offset = header["Segmentation_ReferenceImageExtentOffset"]
    result = re.finditer(r'[\s]', image_offset)
    
    offset_list = []
    xxx = 0
    for m in result:
#        print(m.start(0))
        #print(m.)
        offset_list.append(int(image_offset[xxx:m.start(0)]))
        xxx = m.start(0)+1
    offset_list.append(int(image_offset[xxx:]))
        
 #   print(header["sizes"])
 #   print(header["Segmentation_ReferenceImageExtentOffset"])
    CT_data_fn = ref_fn
    CT_data_orig = nib.load(CT_data_fn)
    
    my_data = CT_data_orig.get_fdata() -CT_data_orig.get_fdata()
    my_data[offset_list[0]:offset_list[0]+header["sizes"][0],offset_list[1]:offset_list[1]+header["sizes"][1], offset_list[2]:offset_list[2] + header["sizes"][2]] = nrrd_data        
    
    seg_resized_nii = nib.Nifti1Image(my_data, CT_data_orig.affine, CT_data_orig.header)
    
    os.makedirs(dest_folder, exist_ok=True)
    nrrd_fn = os.path.basename(nrrd_fn)
    seg_path = dest_folder + '/'+ prefix +'_' + '.nii.gz'
    nib.save(seg_resized_nii, seg_path)
    
