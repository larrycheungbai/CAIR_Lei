#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:47:06 2023

@author: lei
"""


import glob    
import os
import shutil
import pandas as pd
import numpy as np

result_folder = '../results_for_paper/*'


result_list = sorted(glob.glob(result_folder))

target_folder = '../results_for_paper_csv_whole/'

os.makedirs(target_folder, exist_ok=True)
data_empty = {
  'Level':[''],
 'ROI HU':[''],
 'Seg HU':[''],
 'Index':[''],
 'Muscle HU':[''],
 'Muscle CSA (cm^2)':[''],
 'SAT HU':[''],
 'SAT CSA (cm^2)':[''],
 'VAT HU':[''],
 'VAT CSA (cm^2)':[''],
 'IMAT HU':[''],
 'IMAT CSA (cm^2)':['']
 }


df_empty = pd.DataFrame(data_empty)



idx_non_empty = []

clean_name = []
dirty_name = []

vec_L5 = []
vec_L4 = []
vec_L3 = []
vec_L2 = []
vec_L1 = []
vec_T12 = []
vec_sub = []
vec_length = []

count = 0

data_all = {
  'Pst_Name':[''],
  'T12_Level':[''],
 'T12_ROI HU':[''],
 'T12_Seg HU':[''],
 'T12_Index':[''],
 'T12_Muscle HU':[''],
 'T12_Muscle CSA (cm^2)':[''],
 'T12_SAT HU':[''],
 'T12_SAT CSA (cm^2)':[''],
 'T12_VAT HU':[''],
 'T12_VAT CSA (cm^2)':[''],
 'T12_IMAT HU':[''],
 'T12_IMAT CSA (cm^2)':[''],
 'L1_Level':[''],
 'L1_ROI HU':[''],
 'L1_Seg HU':[''],
 'L1_Index':[''],
 'L1_Muscle HU':[''],
 'L1_Muscle CSA (cm^2)':[''],
 'L1_SAT HU':[''],
 'L1_SAT CSA (cm^2)':[''],
 'L1_VAT HU':[''],
 'L1_VAT CSA (cm^2)':[''],
 'L1_IMAT HU':[''],
 'L1_IMAT CSA (cm^2)':[''],
 'L2_Level':[''],
 'L2_ROI HU':[''],
 'L2_Seg HU':[''],
 'L2_Index':[''],
 'L2_Muscle HU':[''],
 'L2_Muscle CSA (cm^2)':[''],
 'L2_SAT HU':[''],
 'L2_SAT CSA (cm^2)':[''],
 'L2_VAT HU':[''],
 'L2_VAT CSA (cm^2)':[''],
 'L2_IMAT HU':[''],
 'L2_IMAT CSA (cm^2)':[''],
 'L3_Level':[''],
 'L3_ROI HU':[''],
 'L3_Seg HU':[''],
 'L3_Index':[''],
 'L3_Muscle HU':[''],
 'L3_Muscle CSA (cm^2)':[''],
 'L3_SAT HU':[''],
 'L3_SAT CSA (cm^2)':[''],
 'L3_VAT HU':[''],
 'L3_VAT CSA (cm^2)':[''],
 'L3_IMAT HU':[''],
 'L3_IMAT CSA (cm^2)':[''],
 'L4_Level':[''],
 'L4_ROI HU':[''],
 'L4_Seg HU':[''],
 'L4_Index':[''],
 'L4_Muscle HU':[''],
 'L4_Muscle CSA (cm^2)':[''],
 'L4_SAT HU':[''],
 'L4_SAT CSA (cm^2)':[''],
 'L4_VAT HU':[''],
 'L4_VAT CSA (cm^2)':[''],
 'L4_IMAT HU':[''],
 'L4_IMAT CSA (cm^2)':[''],
 'L5_Level':[''],
 'L5_ROI HU':[''],
 'L5_Seg HU':[''],
 'L5_Index':[''],
 'L5_Muscle HU':[''],
 'L5_Muscle CSA (cm^2)':[''],
 'L5_SAT HU':[''],
 'L5_SAT CSA (cm^2)':[''],
 'L5_VAT HU':[''],
 'L5_VAT CSA (cm^2)':[''],
 'L5_IMAT HU':[''],
 'L5_IMAT CSA (cm^2)':['']
 }




df_all = pd.DataFrame(data_all)
for idx_result, result_name in enumerate(sorted(result_list)):
    #if count > 10:
    #    break;
        
    #count = count + 1
    if os.listdir(result_name):
        #print(" is not empty, we do not need to do it again ")
        #idx_non_empty.append(idx_result)
        sub_name = os.listdir(result_name)
        xxx = os.path.basename(result_name)
        print(sub_name,xxx)
        #result_dirs = os.listdir(result_name + '/' + sub_name[0])       
        #shutil.copytree(result_name + '/' + 'metrics', target_folder + xxx +'/' + 'metrics',dirs_exist_ok =True)
        data_frame_muscle = pd.read_csv(result_name + '/' +  'metrics' + '/' + 'muscle_adipose_tissue_metrics.csv')
        #vec_L5.append(data_frame['IMAT CSA (cm^2)'][0])
        #vec_L4.append(data_frame['IMAT CSA (cm^2)'][1])
        #vec_L3.append(data_frame['IMAT CSA (cm^2)'][2])
        #vec_L2.append(data_frame['IMAT CSA (cm^2)'][3])
        #vec_L1.append(data_frame['IMAT CSA (cm^2)'][4])
        #vec_T12.append(data_frame['IMAT CSA (cm^2)'][5])
        #print(data_frame_muscle)
        
        data_frame_spine = pd.read_csv(result_name + '/' +  'metrics' + '/' + 'spine_metrics.csv')
        
        df_combined = pd.merge(data_frame_spine, data_frame_muscle, on = 'Level',how='outer') 
        
        
        series_L5 = df_combined[df_combined["Level"].str.contains('L5')]
        if len(series_L5):
            vec_L5.append((series_L5))
        else:
            vec_L5.append(0)
            series_L5 = df_empty
            
        series_L4 = df_combined[df_combined["Level"].str.contains('L4')]
        if len(series_L4):
            vec_L4.append(float(series_L4["IMAT CSA (cm^2)"].iloc[0]))
        else:
            vec_L4.append(0)
            series_L4 = df_empty
            
        series_L3 = df_combined[df_combined["Level"].str.contains('L3')]
        if len(series_L3):
            vec_L3.append(float(series_L3["IMAT CSA (cm^2)"].iloc[0]))
        else:
            vec_L3.append(0)
            series_L3 = df_empty
            
        series_L2 = df_combined[df_combined["Level"].str.contains('L2')]
        if len(series_L2):
            vec_L2.append(float(series_L2["IMAT CSA (cm^2)"].iloc[0]))
        else:
            vec_L2.append(0)
            series_L2 = df_empty
            
        series_L1 = df_combined[df_combined["Level"].str.contains('L1')]
        if len(series_L1):
            vec_L1.append(float(series_L1["IMAT CSA (cm^2)"].iloc[0]))
        else:
            vec_L1.append(0)
            series_L1 = df_empty
        
        series_T12 = df_combined[df_combined["Level"].str.contains('T12')]
        if len(series_T12):
            vec_T12.append(float(series_T12["IMAT CSA (cm^2)"].iloc[0]))
        else:
            vec_T12.append(0)
            series_T12 = df_empty
        
        
        appended_series = series_T12.iloc[0].append(series_L1.iloc[0],ignore_index=True)
        appended_series = appended_series.append(series_L2.iloc[0],ignore_index=True)
        appended_series = appended_series.append(series_L3.iloc[0],ignore_index=True)
        appended_series = appended_series.append(series_L4.iloc[0],ignore_index=True)
        appended_series = appended_series.append(series_L5.iloc[0],ignore_index=True)
        
        #appended_series

        #df_all.insert(df_all.row, appended_series.name, appended_series)
        
        #df_all.loc[:, appended_series.name] = appended_series
        
        yyy = appended_series.values
        row_vals = np.insert(yyy, 0, xxx)
        row1 = pd.Series(row_vals, index=df_all.columns)
        df_all = df_all.append(row1,ignore_index=True)
        
        
df_all.to_csv(target_folder + '/' + 'all_spine_muscle_adipose_tissue_metrics.csv')