import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import io
from bs4 import BeautifulSoup
from functions import lightkurve_plot_axis_on, lightkurve_plot_axis_on_endtofront, lightkurve_plot_axis_on_cutup, split_array, lightkurve_plot_axis_colorful, lightkurve_plot_axis_on_colorful_endtofront
import ast
from PIL import Image

#this should be complete code without any changes needed
#accesses data and then saves the plots


#converts the data in our txt file into a usable list of links
#this is a leftover from the start of the project when the idea of accessing ZTF parquet files directly was entertained
"""with open('file_links_0.txt', mode = 'r') as file:
    # this returns a string
    string_list = file.read()
    #this gets us an actual list with the data being strings
    file_links_0 = string_list.strip('][').split(',')
for i in range(len(file_links_0)):
    #due to formatting we need to remove whitespaces and '
    file_links_0[i] = file_links_0[i].strip(" \'")


with open('file_links_1.txt', mode = 'r') as file:
    string_list = file.read()
    file_links_1 = string_list.strip('][').split(',')
for i in range(len(file_links_1)):
    file_links_1[i] = file_links_1[i].strip(" \'")

file_links = [*file_links_0, *file_links_1]"""


#CODE THAT SAVES CLASS LABELS INTO A CSV FILE
"""#this reads our matched parquet from the supercomputer where we used STILTS into pandas dataframe
directory_match = 'D:\magnetai\\university\\anomalies_oxford\match_parquet\combined_match_glamdring.parquet'
table = pq.read_table(directory_match)
df_match = table.to_pandas()
df_match = df_match.drop(df_match.columns[1:20], axis = 1) #splices our dataframe to get rid of everything besides the labels
df_match = df_match.drop(df_match.columns[-1], axis = 1)
print(df_match.columns.values.tolist())

#this accesses all of our csv files from manual matching using TOPCAT
dir_path='D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_wo_glamd\\'
directory = os.listdir(dir_path)
dataframe_list = []
for file in directory:
    full_path_file = os.path.join(dir_path, file)
    print(full_path_file)
    df = pd.read_csv(full_path_file)
    #print(len(df.columns.values.tolist()))
    if len(df.columns.values.tolist()) != 74:
        z = len(df.columns.values.tolist()) - 73
        df = df.drop(df.columns[-z:], axis=1)
    else:
        df = df.drop(df.columns[-1:], axis=1) #we are now left only with object ids and labels
    df = df.drop(df.columns[1:20], axis=1)
    print(df.columns.values.tolist())
    dataframe_list.append(df)
print('concatenating the tables')
dataframe_list.append(df_match)
dataframe = pd.concat(dataframe_list, ignore_index=True)
#convert every label into 0 or 1 inside our training data so that we could use binary cross entropy
dataframe = dataframe.map(lambda x: 1 if 0.7 < x <= 1 else (0 if x <= 0.7 else x))


#the following code merges all the labels with few examples into one single column called "other"

column_list = dataframe.columns.values.tolist()
merge_list = []
for column in column_list[1:]:
    if dataframe[column].sum() <= 200:
        merge_list.append(column)
dataframe['other'] = dataframe[merge_list].sum(axis=1)
dataframe = dataframe.drop(merge_list, axis=1)
print(dataframe.columns)
print("Our new label number is {}".format(len(dataframe.columns)-1))
for i in range(len(dataframe['other'])):
    if dataframe.loc[i, 'other'] >= 0.7:
        dataframe.loc[i, 'other'] = 1
    else:
        dataframe.loc[i, 'other'] = 0

#part of the code where we modify the ztf_id as per transformations done on the image files
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_endfront'
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_leftright'
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_topbottom'
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_90'
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_180'
#dataframe['objectid'] = dataframe['objectid'].astype(str) + '_270'
#copied_row_list = []
#for i in range(len(dataframe['objectid'])):
#    row_to_copy = dataframe.iloc[i]
#    copies = [row_to_copy.copy() for _ in range(5)]
#    for j, copy in enumerate(copies, start=1):
#        copy['objectid'] = f"{copy['objectid']}_{j}"
#        copied_row_list.append(copy)
    #dataframe.drop(dataframe['objectid'][0])
    #we are not dropping the original rows since not all of the images were cut up
#dataframe = pd.concat([dataframe, pd.DataFrame(copied_row_list)], ignore_index=True)

print(dataframe.columns.values.tolist())
#saving this dataframe into a single file for model training (for model training remove ALL the nonrelevant columns)
dataframe.to_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_270.csv')""" #figure out how to not save indices because it will load them
"""
#CODE FOR MERGING TRAINING CSV'S (we need to get rid of the first column since it is just the row number)
dataframe_1 = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other.csv')
dataframe_endfront = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_endfront.csv')
dataframe_leftright = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_leftright.csv')
dataframe_topbottom = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_topbottom.csv')
dataframe_90 = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_90.csv')
dataframe_180 = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_180.csv')
dataframe_270 = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_270.csv')
dataframe_1 = dataframe_1.drop(dataframe_1.columns[0], axis=1)
dataframe_endfront = dataframe_endfront.drop(dataframe_endfront.columns[0], axis=1)
dataframe_leftright = dataframe_leftright.drop(dataframe_leftright.columns[0], axis=1)
dataframe_topbottom = dataframe_topbottom.drop(dataframe_topbottom.columns[0], axis=1)
dataframe_90 = dataframe_90.drop(dataframe_90.columns[0], axis=1)
dataframe_180 = dataframe_180.drop(dataframe_180.columns[0], axis=1)
dataframe_270 = dataframe_270.drop(dataframe_270.columns[0], axis=1)

dataframe = pd.concat([dataframe_1, dataframe_endfront, dataframe_leftright, dataframe_topbottom, dataframe_90, dataframe_180, dataframe_270], ignore_index=True)
dataframe.to_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_multiplied.csv')
"""

#CODE FOR NUMBER OF EXAMPLES EACH CLASS HAS
"""
examples_per_label = pd.read_csv('D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_training\\training_set_5_other_multiplied.csv')
examples_per_label_cols = examples_per_label.columns.values.tolist()
summ_list = []
for column in examples_per_label_cols[2:]:
    print("label {} has {} examples".format(column, examples_per_label[column].sum()))
    summ_list.append(examples_per_label[column].sum())
print(sorted(summ_list))
"""


#CODE THAT PLOTS THE LIGHT CURVE IMAGES USING FUNCTIONS FROM functions.py
"""
#reads our matched parquet from the supercomputer where we used STILTS into pandas dataframe
directory_match = 'D:\magnetai\\university\\anomalies_oxford\match_parquet\combined_match_glamdring.parquet'
table = pq.read_table(directory_match)
df_match = table.to_pandas()
df_match = df_match.drop(df_match.columns[-1], axis = 1)
#print(df_match.columns.values.tolist())

#accesses all of our csv files from manual matching using TOPCAT
dir_path='D:\magnetai\\university\\anomalies_oxford\match_parquet\csv_wo_glamd\\'
directory = os.listdir(dir_path)
dataframe_list = []
for file in directory:
    full_path_file = os.path.join(dir_path, file)
    print(full_path_file)
    df = pd.read_csv(full_path_file)
    #print(len(df.columns.values.tolist()))
    if len(df.columns.values.tolist()) != 74:
        z = len(df.columns.values.tolist()) - 73
        df = df.drop(df.columns[-z:], axis=1)
    else:
        df = df.drop(df.columns[-1:], axis=1)
    print(df.columns.values.tolist())
    dataframe_list.append(df)
print('concatenating the tables')
dataframe_list.append(df_match) #the combined_match_glamdring.parquet is appended at the end
dataframe = pd.concat(dataframe_list, ignore_index=True)

print(dataframe.columns.values.tolist())
print(dataframe)


#for plot saving need to only remove the last few columns as specified above (i.e. we need the [1:20] columns)

#choose which kind of plot you want to save
#lightkurve_plot_axis_on(dataframe, save_location = 'lightkurve_plots_match')
#lightkurve_plot_axis_on_endtofront(dataframe, save_location = 'lightkurve_plots_match_endfront')
#lightkurve_plot_axis_on_cutup(dataframe, save_location = 'lightkurve_plots_match_cutup')
#lightkurve_plot_axis_colorful(dataframe, save_location = 'lightkurve_plots_match_colorful\lightkurve_plots_match_colorful')
#lightkurve_plot_axis_on_colorful_endtofront(dataframe, save_location = 'lightkurve_plots_match_colorful\lightkurve_plots_match_colorful_endfront')
"""

#CODE FOR IMAGE AUGMENTATION (rotation, flipping and so on)
"""
img_dir = 'C:\\Users\zaviv\PycharmProjects\kuro\lightkurve_plots_match_colorful\lightkurve_plots_match_colorful'
os.listdir(img_dir)
for file in os.listdir(img_dir):
    filename = os.path.join(img_dir, file)
    print(file) #check if it works

    img = Image.open(filename)
    file = file[:-4]
    print(file)
    # Flip the image
    img_leftright = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_topbottom = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_90 = img.transpose(Image.ROTATE_90)
    img_180 = img.transpose(Image.ROTATE_180)
    img_270 = img.transpose(Image.ROTATE_270)
    
    img_leftright.save('lightkurve_plots_match_colorful\lightkurve_plots_match_colorful_leftright\{}_leftright.jpg'.format(file))
    img_topbottom.save('lightkurve_plots_match_colorful\lightkurve_plots_match_colorful_topbottom\{}_topbottom.jpg'.format(file))
    img_90.save('lightkurve_plots_match_colorful\lightkurve_plots_colorful_match_90\{}_90.jpg'.format(file))
    img_180.save('lightkurve_plots_match_colorful\lightkurve_plots_colorful_match_180\{}_180.jpg'.format(file))
    img_270.save('lightkurve_plots_match_colorful\lightkurve_plots_colorful_match_270\{}_270.jpg'.format(file))"""

