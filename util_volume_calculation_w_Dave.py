#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:25:29 2023

@author: Lei Zhang
"""
import nibabel as nib
import numpy as np

#Input is the nii file name, the output is the quantitative results of the mask (0, as background, non-zeros as mask labels)
#mask_volume is the count of voxels in the volume
#vol_ml is the volume measured in ml
#the volume_norm is calculated from the header of the nii file to normalize the mask volume into ml.

def vol_in_ml(abs_nii_fn):
    mask_volume_test =nib.load(abs_nii_fn)
    header_gtest = mask_volume_test.header 
    
    hdr = header_gtest
    
    
    print(header_gtest['xyzt_units'], hdr.get_xyzt_units()) 
    
    x_dim = header_gtest['pixdim'][1]
    y_dim = header_gtest['pixdim'][2]
    z_dim = header_gtest['pixdim'][3]
    
    
    #volume_norm is calculate based on https://brainder.org/2012/09/23/the-nifti-file-format/    #
    volume_norm  = x_dim * y_dim * z_dim
    
    mask_volume =nib.load(abs_nii_fn).get_fdata()
    mask_volume = np.sum(mask_volume>0)
    
    print('x_dim, y_dim, z_dim, volume_norm ', x_dim, y_dim, z_dim, volume_norm)
    
    #from mm to ml
    vol_ml = mask_volume * volume_norm /1000  
    
    return mask_volume, vol_ml, volume_norm




#vols_num, volume_ml, norm_val =vol_in_ml('/media/ubuntu/SATA_WDC/2023_WBC_dataset_tranche/Result_selected_HEMOTHORAX/TraumaBody_15_1_1_CT_HEADBRAIN_WO_CON_DE_BODY_4.0_Br40_3_F_0.7_13.nii.gz')


