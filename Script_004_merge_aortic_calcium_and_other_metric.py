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
import pandas as pd

target_folder = '../results_for_paper_csv_whole/'


# Load the first CSV file into a DataFrame
df1 = pd.read_csv(target_folder + '/' + 'aortic_calcium.csv')

# Load the second CSV file into another DataFrame
df2 = pd.read_csv(target_folder + '/' + 'all_spine_muscle_adipose_tissue_metrics.csv')

# Merge the DataFrames using the common key
merged_df = pd.merge(df1, df2, on='Pst_Name', how='outer')

# Display the merged DataFrame
print(merged_df)

        
merged_df.to_csv(target_folder + '/' + 'all_spine_muscle_adipose_tissue_aortic_calcium.csv')
