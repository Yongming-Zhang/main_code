import json
import os
import glob 
import cv2
from mitok.image.cv_gpu import label, pairwise_distances
import numpy as np
import struct
import SimpleITK as sitk
import torch
from mitok.utils.mdicom import SERIES
import matplotlib.pyplot as plt
import nibabel as nib

def read_centerline(cpr_centerline_paths):
    with open(cpr_centerline_paths, mode='r') as f:
        centerline = json.load(f)
    return centerline

patient = '1022837'
series = '62D38FAD_CTA'
vessel_centerline_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'
plaque_centerline_file = '/mnt/users/code/test/cerebral_1022837_1.json'
plaque_mask_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'

all_vessel_id = []
all_centerline_range = []
with open(plaque_centerline_file, 'r') as f:
    datas = json.load(f)
    plaque_datas = datas["plaque"]
    for plaque_data in plaque_datas:
        plaque_type = plaque_data['plaque_type']
        if plaque_type == "calcified":
            merged_centerline_range = plaque_data['merged_centerline_range']
            all_centerline_range.append(merged_centerline_range)
            vessel_id = plaque_data['vessel_id']
            all_vessel_id.append(vessel_id)
print(all_vessel_id)
print(all_centerline_range)

cpr_plaque_mask_path = glob.glob(os.path.join(vessel_centerline_path, patient, '*', series, 'mask_source', 'mask_plaque.nii.gz'))[0]
plaque_data = nib.load(cpr_plaque_mask_path)
plaque_data = plaque_data.get_fdata()
print(plaque_data.shape)
plaque_region_data, plaque_label_area = label(plaque_data, 0, to_numpy=True, connectivity=1)
plaque_label_area = dict(map(lambda x: (x.label, x.area), plaque_label_area))
print(plaque_label_area)

plaque_calcified = [] 
for i in range(len(all_vessel_id)):
    coro_id = all_vessel_id[i]
    print('coro_id', coro_id)

    #对中心线2D坐标转成3D坐标
    axis_centerline_paths = glob.glob(os.path.join(vessel_centerline_path, patient, '*', series, 'centerline'))[0]
    axis_centerline_paths = os.path.join(axis_centerline_paths, coro_id+'.3d')
    all_axis_centerlines = read_centerline(axis_centerline_paths)

    axis_centerlines = []
    for k in range(all_centerline_range[i][0], all_centerline_range[i][1]):
        axis_centerlines.append(all_axis_centerlines['points'][k])

    for plaque_num, plaque_volume in plaque_label_area.items():
        plaque_index = np.argwhere(plaque_region_data==plaque_num)
        #print('plaque_index', plaque_index)
        #print('axis_centerlines', axis_centerlines)
        distance = pairwise_distances(plaque_index, device=torch.device("cuda"), y=axis_centerlines)
        #print(distance)
        if np.min(distance) < 10:
            if plaque_num not in plaque_calcified:
                plaque_calcified.append(plaque_num)
                break
print(plaque_calcified)

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    #plt.hist(myList,100)
    bins = range(Xmin, Xmax+1, 10)
    n, bins, patches = plt.hist(x=myList, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.xlabel(Xlabel)
    #plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    #plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

plaque_data = np.transpose(plaque_data, (2,1,0))
plaque_region_data, plaque_label_area = label(plaque_data, 0, to_numpy=True, connectivity=1)
plaque_label_area = dict(map(lambda x: (x.label, x.area), plaque_label_area))
print(plaque_label_area)
dcm_folder = glob.glob(os.path.join(plaque_mask_path, patient, '*', series.split('_CTA')[0]))[0]
series = SERIES(series_path=dcm_folder, strict_check_series=True)
img_tensor_int16 = series.get_image_tensor_int16()
for plaque_num, plaque_volume in plaque_label_area.items():
    if plaque_num in plaque_calcified:
        data_pixel = img_tensor_int16[plaque_region_data==plaque_num]
        print(data_pixel)
        min_value = min(data_pixel)
        max_value = max(data_pixel)
        draw_hist(data_pixel,'Hist','Pixel','Number',min_value-10,max_value+10,0,10)   # 直方图展示