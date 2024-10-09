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
for idx, image_fn in enumerate(sorted(image_list)):      
    #if idx > 5:
    #    break
    image_base_fn = os.path.basename(image_fn)
#    vols_num, volume_ml, norm_val    
    gt_mask_vols_num, gt_volume_ml, gt_norm_val = vol_in_ml(image_fn)
    
    print('gt stats ', vol_in_ml(image_fn))
    gt_element = nib.load(root_folder + image_base_fn)
    gt_data = gt_element.get_fdata()
    #    predict_seg = predict_element['seg']
    all_labels = np.unique(gt_data.flatten())
    print("all ", all_labels)   
    sum_all_label_volume_num = 0 
    sum_all_label_volume_in_ml = 0
    
    label_vect_volume_num = [0 for i in range(8)]
    label_vect_volume_ml  = [0 for i in range(8)]
    #print(label_vect)
    for cur_label in all_labels:
    	print(cur_label)
    	if cur_label >0:
    		cur_label_mask = gt_data[gt_data==cur_label]
    		cur_label_volume_num = np.sum(cur_label_mask>0)   		
    		cur_label_vol_ml = cur_label_volume_num * gt_norm_val /1000 
    		#mask_volume = np.sum(mask_volume>0)
    		print('cur_label_volume_num', cur_label_volume_num, cur_label_vol_ml)
    		sum_all_label_volume_num += cur_label_volume_num
    		sum_all_label_volume_in_ml+=cur_label_vol_ml
    		cur_label = int(cur_label)
    		label_vect_volume_num[cur_label] = cur_label_volume_num
    		label_vect_volume_ml[cur_label]  = cur_label_vol_ml
    		    
    print(gt_mask_vols_num, gt_volume_ml, sum_all_label_volume_num, sum_all_label_volume_in_ml)
    #
    patient_idx = image_base_fn[:-7]
    
    nnunet_image_fn = results_folder+image_base_fn
    nnunet_mask_vols_num, nnunet_volume_ml, nnunet_norm_val = vol_in_ml(nnunet_image_fn)
    
    print('nnunet stats ', vol_in_ml(nnunet_image_fn))
    nnunet_element = nib.load(nnunet_image_fn)
    nnunet_data = nnunet_element.get_fdata()
    #    predict_seg = predict_element['seg']
    all_nnunet_labels = np.unique(nnunet_data.flatten())
    print("all ", all_nnunet_labels)   
    sum_all_nnunet_volume_num = 0 
    sum_all_nnunet_volume_in_ml = 0
    
    nnunet_vect_volume_num = [0 for i in range(8)]
    nnunet_vect_volume_ml  = [0 for i in range(8)]
    #print(label_vect)
    for cur_label in all_nnunet_labels:
    	print(cur_label)
    	if cur_label >0:
    		cur_label_mask = nnunet_data[nnunet_data==cur_label]
    		cur_label_volume_num = np.sum(cur_label_mask>0)   		
    		cur_label_vol_ml = cur_label_volume_num * nnunet_norm_val /1000 
    		#mask_volume = np.sum(mask_volume>0)
    		print('cur_label_volume_num', cur_label_volume_num, cur_label_vol_ml)
    		sum_all_nnunet_volume_num += cur_label_volume_num
    		sum_all_nnunet_volume_in_ml+=cur_label_vol_ml
    		cur_label = int(cur_label)
    		nnunet_vect_volume_num[cur_label] = cur_label_volume_num
    		nnunet_vect_volume_ml[cur_label]  = cur_label_vol_ml


   # cur_data_row = #(patient_idx, gt_mask_vols_num, label_vect_volume_num[1], label_vect_volume_num[2], label_vect_volume_num[3], label_vect_volume_num[4], label_vect_volume_num[5], label_vect_volume_num[6],
                   # label_vect_volume_num[7],
    #cur_data_row =   (patient_idx,gt_volume_ml, label_vect_volume_ml[1], label_vect_volume_ml[2], label_vect_volume_ml[3], label_vect_volume_ml[4], label_vect_volume_ml[5], label_vect_volume_ml[6],
    #                label_vect_volume_ml[7],
    #                nnunet_mask_vols_num, nnunet_vect_volume_num[1], nnunet_vect_volume_num[2], nnunet_vect_volume_num[3], nnunet_vect_volume_num[4], nnunet_vect_volume_num[5], nnunet_vect_volume_num[6],
     #               nnunet_vect_volume_num[7],
    cur_data_row =   (patient_idx,gt_volume_ml, label_vect_volume_ml[1], label_vect_volume_ml[2], label_vect_volume_ml[3], label_vect_volume_ml[4], label_vect_volume_ml[5], label_vect_volume_ml[6],
                     label_vect_volume_ml[7],
                    nnunet_volume_ml, nnunet_vect_volume_ml[1], nnunet_vect_volume_ml[2], nnunet_vect_volume_ml[3], nnunet_vect_volume_ml[4], nnunet_vect_volume_ml[5], nnunet_vect_volume_ml[6],
                    nnunet_vect_volume_ml[7])
    
    
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
                           'automated_label_7_vol_ml'])

#Relative Volume Difference (RVD):
#The formula for RVD between manual (ground truth) and automated volumes for a given label is:
#𝑅𝑉𝐷=∣𝑉manual−𝑉automated∣ / 𝑉manual
#Volume Similarity (VS):
#VS=1− (∣Vmanual −Vautomated∣/(Vmanual+Vautomated))
​
# Create columns for RVD and VS for each label (1 through 7)
for label in range(1, 8):
    manual_col = f'manual_label_{label}_vol_ml'
    automated_col = f'automated_label_{label}_vol_ml'
    
    # RVD calculation
    df_multiple[f'RVD_label_{label}'] = np.abs(df_multiple[manual_col] - df_multiple[automated_col]) / df_multiple[manual_col]
    
    # VS calculation
    df_multiple[f'VS_label_{label}'] = 1 - (np.abs(df_multiple[manual_col] - df_multiple[automated_col]) / 
                                             (df_multiple[manual_col] + df_multiple[automated_col]))

# Show the first few rows to check the results
print(df_multiple.head())


df_multiple.to_csv('manual_vs_automated_vol_in_ml_RVD_VS.csv', index=False)
