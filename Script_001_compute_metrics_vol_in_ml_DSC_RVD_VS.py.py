import os 
import glob
import nibabel as nib
root_folder  = '../labelsTr/'
results_folder = '../validation_five_fold_results/'
#gt image
image_list = glob.glob(root_folder + '*.nii.gz')

import numpy as np
from util_volume_calculation_w_Dave import vol_in_ml
#import 
data_tuples = []

def calculate_dsc(gt, pred):
    gt = np.array(gt) > 0  # Convert to binary mask
    pred = np.array(pred) > 0  # Convert to binary mask
    intersection = np.sum(gt & pred)
    return (2 * intersection) / (np.sum(gt) + np.sum(pred)) if (np.sum(gt) + np.sum(pred)) > 0 else 0.0


for idx, image_fn in enumerate(sorted(image_list)):      
    #if idx > 3:  # Limit to first 6 images
    #    break
    
    image_base_fn = os.path.basename(image_fn)
    
    # Ground truth data (GT)
    gt_mask_vols_num, gt_volume_ml, gt_norm_val = vol_in_ml(image_fn)
    print('Ground Truth Stats: ', gt_mask_vols_num, gt_volume_ml, gt_norm_val)
    
    gt_element = nib.load(root_folder + image_base_fn)
    gt_data = gt_element.get_fdata()
    all_gt_labels = np.unique(gt_data.flatten())
    print("Ground Truth Labels: ", all_gt_labels)
    
    # Initialize GT volume accumulators
    sum_all_gt_volume_num = 0
    sum_all_gt_volume_in_ml = 0
    
    label_vect_volume_num = [0 for i in range(8)]
    label_vect_volume_ml  = [0 for i in range(8)]
    #label_mask_volume_vec = []
    label_mask_volume_vec = [None for _ in range(8)]

    for cur_label in all_gt_labels:
        if cur_label > 0:  # Exclude background
            cur_label_mask = (gt_data == cur_label).astype(int)
            label_mask_volume_vec[int(cur_label)] = cur_label_mask
            
            cur_label_volume_num = np.sum(cur_label_mask)  # Voxel count
            cur_label_volume_in_ml = cur_label_volume_num * gt_norm_val / 1000  # Volume in ml
            
            print(f'GT Label {cur_label}: {cur_label_volume_num} voxels, {cur_label_volume_in_ml:.2f} ml')
            
            sum_all_gt_volume_num += cur_label_volume_num
            sum_all_gt_volume_in_ml += cur_label_volume_in_ml
            
            cur_label = int(cur_label)
            label_vect_volume_num[cur_label] = cur_label_volume_num
            label_vect_volume_ml[cur_label]  = cur_label_volume_in_ml
    print()

    # Now for nnU-Net data
    nnunet_image_fn = results_folder + image_base_fn
    nnunet_mask_vols_num, nnunet_volume_ml, nnunet_norm_val = vol_in_ml(nnunet_image_fn)
    print('nnU-Net Stats: ', nnunet_mask_vols_num, nnunet_volume_ml, nnunet_norm_val)
    
    nnunet_element = nib.load(nnunet_image_fn)
    nnunet_data = nnunet_element.get_fdata()
    all_nnunet_labels = np.unique(nnunet_data.flatten())
    print("nnU-Net Labels: ", all_nnunet_labels)
    
    # Initialize nnU-Net volume accumulators
    sum_all_nnunet_volume_num = 0
    sum_all_nnunet_volume_in_ml = 0
    
    nnunet_vect_volume_num = [0 for i in range(8)]
    nnunet_vect_volume_ml  = [0 for i in range(8)]
    #nnunet_mask_volume_vec = []
    
    # Initialize vectors to store label masks and set the size to 8, where index 0 is unused
    
    nnunet_mask_volume_vec = [None for _ in range(8)]

    for cur_label in all_nnunet_labels:
        if cur_label > 0:  # Exclude background
            cur_label_mask = (nnunet_data == cur_label).astype(int)
            nnunet_mask_volume_vec[int(cur_label)] = cur_label_mask
            
            cur_label_volume_num = np.sum(cur_label_mask)  # Voxel count
            cur_label_volume_in_ml = cur_label_volume_num * nnunet_norm_val / 1000  # Volume in ml
            
            print(f'nnU-Net Label {cur_label}: {cur_label_volume_num} voxels, {cur_label_volume_in_ml:.2f} ml')
            
            sum_all_nnunet_volume_num += cur_label_volume_num
            sum_all_nnunet_volume_in_ml += cur_label_volume_in_ml
            
            cur_label = int(cur_label)
            nnunet_vect_volume_num[cur_label] = cur_label_volume_num
            nnunet_vect_volume_ml[cur_label]  = cur_label_volume_in_ml

    # Calculate Dice Similarity Coefficients (DSC)
    dsc_vect = [0 for i in range(8)]
    for i in range(1, 8):
        if label_mask_volume_vec[i] is not None and nnunet_mask_volume_vec[i] is not None:
            print(f'Label {i} - GT Mask Shape: {label_mask_volume_vec[i].shape}, nnU-Net Mask Shape: {nnunet_mask_volume_vec[i].shape}')
            
            print(f'Label {i} - GT Mask Shape: {label_vect_volume_num[i]}, nnU-Net Mask Shape: {nnunet_vect_volume_num[i]}')
            
            #print(f'nnunet_vect_volume_ml {i} - GT Mask Shape: {label_mask_volume_vec[i].shape}, nnU-Net Mask Shape: {nnunet_mask_volume_vec[i].shape}')
            
            dsc_vect[i] = calculate_dsc(label_mask_volume_vec[i], nnunet_mask_volume_vec[i])
            print(f'DSC for Label {i}: {dsc_vect[i]:.4f}')
        else:
            print(f'No matching label {i} in both GT and nnU-Net data.')
    	
   # cur_data_row = #(patient_idx, gt_mask_vols_num, label_vect_volume_num[1], label_vect_volume_num[2], label_vect_volume_num[3], label_vect_volume_num[4], label_vect_volume_num[5], label_vect_volume_num[6],
                   # label_vect_volume_num[7],
    #cur_data_row =   (patient_idx,gt_volume_ml, label_vect_volume_ml[1], label_vect_volume_ml[2], label_vect_volume_ml[3], label_vect_volume_ml[4], label_vect_volume_ml[5], label_vect_volume_ml[6],
    #                label_vect_volume_ml[7],
    #                nnunet_mask_vols_num, nnunet_vect_volume_num[1], nnunet_vect_volume_num[2], nnunet_vect_volume_num[3], nnunet_vect_volume_num[4], nnunet_vect_volume_num[5], nnunet_vect_volume_num[6],
     #               nnunet_vect_volume_num[7],
    #cur_data_row =   (patient_idx,gt_volume_ml, label_vect_volume_ml[1], label_vect_volume_ml[2], label_vect_volume_ml[3], label_vect_volume_ml[4], label_vect_volume_ml[5], label_vect_volume_ml[6],
    #                 label_vect_volume_ml[7],
    #                nnunet_volume_ml, nnunet_vect_volume_ml[1], nnunet_vect_volume_ml[2], nnunet_vect_volume_ml[3], nnunet_vect_volume_ml[4], nnunet_vect_volume_ml[5], nnunet_vect_volume_ml[6],
    #                nnunet_vect_volume_ml[7],dsc_vec[1], dsc_vec[2], dsc_vec[3], dsc_vec[4], dsc_vec[5], dsc_vec[6],
    #                dsc_vec[7])
    patient_idx = image_base_fn[:-7]
                    
    cur_data_row = (
    patient_idx,  # Index for the current patient
    # Ground truth volumes (in ml)
    gt_volume_ml, 
    label_vect_volume_ml[1], label_vect_volume_ml[2], label_vect_volume_ml[3], 
    label_vect_volume_ml[4], label_vect_volume_ml[5], label_vect_volume_ml[6], 
    label_vect_volume_ml[7],

    # nnU-Net volumes (in ml)
    nnunet_volume_ml, 
    nnunet_vect_volume_ml[1], nnunet_vect_volume_ml[2], nnunet_vect_volume_ml[3], 
    nnunet_vect_volume_ml[4], nnunet_vect_volume_ml[5], nnunet_vect_volume_ml[6], 
    nnunet_vect_volume_ml[7],

    # Dice Similarity Coefficients (DSC) for each label
    dsc_vect[1], dsc_vect[2], dsc_vect[3], dsc_vect[4], 
    dsc_vect[5], dsc_vect[6], dsc_vect[7])
    
    
    
    #results_nnunet = nib.load(results_folder + image_base_fn)
    data_tuples.append(cur_data_row)
    print(idx, image_base_fn)
    # to compute the 1 specific value 's DVS and other's we can make the others zero.
    # we make a deep copy.
    

import pandas as pd
    
df_multiple = pd.DataFrame(data_tuples, columns=['PST_ID', 'manual_total_vol_ml', 'manual_label_1_vol_ml', 'manual_label_2_vol_ml',
                           'manual_label_3_vol_ml', 'manual_label_4_vol_ml',
                           'manual_label_5_vol_ml', 'manual_label_6_vol_ml',
                           'manual_label_7_vol_ml','automated_total_vol_ml', 'automated_label_1_vol_ml', 'automated_label_2_vol_ml',
                           'automated_label_3_vol_ml', 'automated_label_4_vol_ml',
                           'automated_label_5_vol_ml', 'automated_label_6_vol_ml',
                           'automated_label_7_vol_ml','dsc_label_1', 'dsc_label_2',
                           'dsc_label_3', 'dsc_label_4',
                           'dsc_label_5', 'dsc_label_6',
                           'dsc_label_7'])

#Relative Volume Difference (RVD):
#The formula for RVD between manual (ground truth) and automated volumes for a given label is:
#𝑅𝑉𝐷=∣𝑉manual−𝑉automated∣ / 𝑉manual
#Volume Similarity (VS):
#VS=1− (∣Vmanual −Vautomated∣/(Vmanual+Vautomated))
# Create columns for RVD and VS for each label (1 through 7)
for label in range(1, 8):
    manual_col = f'manual_label_{label}_vol_ml'
    automated_col = f'automated_label_{label}_vol_ml'
    
    # RVD calculation (handle division by zero)
    df_multiple[f'RVD_label_{label}'] = np.where(
        df_multiple[manual_col] == 0, 0, 
        np.abs(df_multiple[manual_col] - df_multiple[automated_col]) / df_multiple[manual_col]
    )
    
    # VS calculation (ensure VS stays within [0, 1])
    df_multiple[f'VS_label_{label}'] = np.where(
        (df_multiple[manual_col] + df_multiple[automated_col]) == 0, 0,
        1 - (np.abs(df_multiple[manual_col] - df_multiple[automated_col]) / 
            (df_multiple[manual_col] + df_multiple[automated_col]))
    )
    df_multiple[f'VS_label_{label}'] = df_multiple[f'VS_label_{label}'].clip(lower=0)  # Ensure non-negative VS

# Show the first few rows to check the results

# Calculate mean and standard deviation for each column (excluding 'PST_ID')
mean_row = df_multiple.mean(numeric_only=True).to_frame().T
std_row = df_multiple.std(numeric_only=True).to_frame().T

# Add row labels for mean and std
mean_row['PST_ID'] = 'Mean'
std_row['PST_ID'] = 'STD'

# Append mean and std rows to the DataFrame
df_multiple = pd.concat([df_multiple, mean_row, std_row], ignore_index=True)

# Show the first few rows to check the results
print(df_multiple.head())

# Save to CSV
df_multiple.to_csv('manual_vs_automated_vol_in_ml_DSC_RVD_VS.csv', index=False)
