import matplotlib.pyplot as plt
#show center slices of a numpy 3D volume
from skimage.util import montage as montage2d
from warnings import warn
import numpy as np
import pydicom
from PIL import ImageTk, Image
#https://pycad.co/how-to-convert-a-dicom-image-into-jpg-or-png/
def convert_dcm_2_png(name):
    im = pydicom.dcmread(name)
    im = im.pixel_array.astype(float)
    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels
    final_image = Image.fromarray(final_image)
    return final_image
    

def is_nan_in_this_array(inputArr):
    #nan added any number is still nan      
    sum_arr = np.sum(inputArr)
    return np.isnan(sum_arr)
    

def is_inf_in_this_array(inputArr):
    #inf added any number is still inf 
    sum_arr = np.sum(inputArr)
    return np.isinf(sum_arr)
    

def montage_nd(in_img):
    if len(in_img.shape)>3:
        return montage2d(np.stack([montage_nd(x_slice) for x_slice in in_img],2))
    elif len(in_img.shape)==3:
        return montage2d(in_img)
    else:
        warn('Input less than 3d image, returning original', RuntimeWarning)
        return in_img
#  slice_idx, x, y 
def show_montage_slices(in_img, fig_title=""):
    plt.imshow(montage_nd(in_img), cmap = 'gray')
    if len(fig_title):
        plt.title(fig_title)
    plt.axis('off')
    plt.show()
    
    
def show_x_ax_center_slices(data_vol, save_png):
    x_dim, y_dim, z_dim = data_vol.shape
    #print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)    
    #return data_vol[x_center_idx,:,:]
    plt.imshow(data_vol[x_center_idx,:,:],cmap='gray')
    plt.axis('off')
    plt.savefig(save_png,bbox_inches='tight')
    
    
    
def show_center_slices(data_vol):
    x_dim, y_dim, z_dim = data_vol.shape
    print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)
    y_center_idx = int(y_dim/2)
    z_center_idx = int(z_dim/2)    
    fix, ax = plt.subplots(1,3, figsize=(10,20))
    ax[0].imshow(data_vol[x_center_idx,:,:],cmap='gray')    
    ax[1].imshow(data_vol[:,y_center_idx,:],cmap='gray')    
    ax[2].imshow(data_vol[:,:,z_center_idx],cmap='gray')    
        
def show_center_slices_mni(data_vol):
    x_dim, y_dim, z_dim = data_vol.shape
    print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)
    y_center_idx = int(y_dim/2)
    z_center_idx = int(z_dim/2)    
    fix, ax = plt.subplots(1,3, figsize=(10,20))
    ax[0].imshow( np.fliplr(np.rot90(data_vol[x_center_idx,:,:], 1)),cmap='gray')    
    ax[1].imshow(np.rot90(data_vol[:,y_center_idx,:], 1),cmap='gray')    
    ax[2].imshow(np.rot90(data_vol[:,:,z_center_idx], -1),cmap='gray')    
    return fix
   
#to nifity space
#reference_vol is a nifity object which can be obtained by using:
#reference_vol = load_img('./Data/wmeanCBF_0_ASLflt_rAD02_RP1_pCASL_S1.nii')
#the np_data is the numpy ndarray (usually 3D) 
#dest_fn is the name for the nii file
#by Lei Zhang 
#University of Maryland School of Medicine.
from nilearn.image import new_img_like, load_img, get_data
def save_np_to_nii(reference_vol, np_data, dest_fn):
    cur_data_nii = new_img_like(reference_vol, np_data)   
    cur_data_nii.to_filename(dest_fn)    

from scipy import ndimage, misc
def resize_3d_npy(atlas_vol, xx_local, yy_local, zz_local):
    (xx_mov,yy_mov,zz_mov)=atlas_vol.shape
    atlas_vol = ndimage.zoom(atlas_vol,(xx_local/xx_mov, yy_local/yy_mov, zz_local/zz_mov))
    return atlas_vol

def get_center_slices(data_vol):
    x_dim, y_dim, z_dim = data_vol.shape
    print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)
    y_center_idx = int(y_dim/2)
    z_center_idx = int(z_dim/2)    
    x_center_img = data_vol[x_center_idx,:,:] 
    y_center_img = data_vol[:,y_center_idx,:]
    z_center_img = data_vol[:,:,z_center_idx]    
    return x_center_img, y_center_img, z_center_img


import numpy as np
def normalize_3d_npy(org_3d_data):    
    max_intensity = org_3d_data.max()
    min_intensity = org_3d_data.min()        
    flatten_3d_data = org_3d_data.flatten()
    top_part = flatten_3d_data - min_intensity
    bottom_part = max_intensity - min_intensity
    bottom_part = np.maximum(bottom_part, np.finfo(float).eps)  # add epsilon.
    norm_3d_data = top_part /bottom_part
    norm_3d_data = norm_3d_data.reshape(org_3d_data.shape)    
    return norm_3d_data
    

def show_and_return_center_slices(data_vol):
    x_dim, y_dim, z_dim = data_vol.shape
    print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)
    y_center_idx = int(y_dim/2)
    z_center_idx = int(z_dim/2)    
    fix, ax = plt.subplots(1,3, figsize=(10,20))
    ax[0].imshow(data_vol[x_center_idx,:,:],cmap='gray')    
    ax[1].imshow(data_vol[:,y_center_idx,:],cmap='gray')    
    ax[2].imshow(data_vol[:,:,z_center_idx],cmap='gray')    
    return fix

def show_and_return_center_slices_mni(data_vol):
    x_dim, y_dim, z_dim = data_vol.shape
    print(x_dim, y_dim, z_dim)
    x_center_idx = int(x_dim/2)
    y_center_idx = int(y_dim/2)
    z_center_idx = int(z_dim/2)    
    fix, ax = plt.subplots(1,3, figsize=(10,20))
    ax[0].imshow( np.fliplr(np.rot90(data_vol[x_center_idx,:,:], 1)),cmap='gray')    
    ax[1].imshow(np.rot90(data_vol[:,y_center_idx,:], 1),cmap='gray')    
    ax[2].imshow(np.rot90(data_vol[:,:,z_center_idx], -1),cmap='gray')    
    return fix

from nilearn.image import new_img_like
#template_vol is a existing .nii file with nii header information
#cur_image is the npy array
#dest_fn is the filename to save the created .nii object
def create_nii_from_npy_with_template(template_vol, cur_image, dest_fn):
    cur_data_nii = new_img_like(template_vol,cur_image)   
    cur_data_nii.to_filename(dest_fn)       
