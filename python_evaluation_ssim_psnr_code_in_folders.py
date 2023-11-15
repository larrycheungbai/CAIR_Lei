import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
result_folder = '../moco_exp_20_s8_clearn_V3_120k/';
input_folder = '../../Experiment20/test_folder_exp_20_s8/corrupt_image/data/'
gt_folder = '../../Experiment20/test_folder_exp_20_s8/clean_image/data/'
fn_out = 'ssim_psnr_restormer_V3_120K_clean_python.txt'
fn_out_ssim_image = 'Exp_20_s8_restormer_V3_120K_clean_python.txt'

vec_ssim = []
vec_psnr = []

with open(fn_out_ssim_image, 'w') as f_out_ssim:
    image_list = os.listdir(input_folder)[2:]

    for img_fn in image_list:
        print(f'total {len(image_list)} left {len(image_list) - image_list.index(img_fn)}')
        
        input_file_path = os.path.join(input_folder, img_fn)
        result = cv2.imread(os.path.join(result_folder, img_fn))
        pred_clean = result
        gt_image = cv2.imread(os.path.join(gt_folder, img_fn))

        ssim_value, _ = ssim(pred_clean[:,:,0], gt_image[:,:,0], full=True)
        psnr_value = cv2.PSNR(pred_clean[:,:,0], gt_image[:,:,0])

        vec_ssim.append(ssim_value)
        vec_psnr.append(psnr_value)

        f_out_ssim.write(f'{img_fn}  {ssim_value:10.5f}  {psnr_value:10.5f}\n')

with open(fn_out, 'w') as f_results:
    f_results.write(f'mean ssim {np.mean(vec_ssim):10.5f}, std ssim {np.std(vec_ssim):10.5f}\n')
    f_results.write(f'mean psnr {np.mean(vec_psnr):10.5f}, std psnr {np.std(vec_psnr):10.5f}\n')

debug_end = 0