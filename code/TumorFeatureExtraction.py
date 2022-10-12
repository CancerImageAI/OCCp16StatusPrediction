# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:44:09 2019

@author: PC
"""

import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import measure


def Img_Normalization(Image_Orig):
    Image_array = sitk.GetArrayFromImage(Image_Orig)
    min_ImgValue = Image_array.min()
    max_ImgValue = Image_array.max()
    ImgRange = max_ImgValue-min_ImgValue
    min_NewValue = 0
    max_NewValue = 1200
    NewRange = max_NewValue-min_NewValue
    Img_array = ((Image_array-min_ImgValue)/ImgRange)*NewRange+min_NewValue
    Img = sitk.GetImageFromArray(Img_array.astype(int))
    Img.SetDirection(Image_Orig.GetDirection())
    Img.SetOrigin(Image_Orig.GetOrigin())
    Img.SetSpacing(Image_Orig.GetSpacing())
#    Img.CopyInformation(Image_Orig)
    return Img
    
def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    start = time.clock()
    warnings.simplefilter('ignore')

    Neg_Path = '../negative'
    Pos_Path = '../positive'
    Pos_Lists = os.listdir(Pos_Path)
    Neg_Lists = os.listdir(Neg_Path)
    
    T1_ROIFeature = []
    T2_ROIFeature = []
    T1_LNFeature = []
    T2_LNFeature = []
    for Pos_Case in tqdm(Pos_Lists):
        sleep(0.01)
        T1_ImgPath = Pos_Path+'/'+Pos_Case+'/'+'T1_'+Pos_Case+'.nii'
        T2_ImgPath = Pos_Path+'/'+Pos_Case+'/'+'T2_'+Pos_Case+'.nii'
        T1_roiMaskPath = Pos_Path+'/'+Pos_Case+'/'+'roi_'+Pos_Case+'.nii'
        T1_lnMaskPath = Pos_Path+'/'+Pos_Case+'/'+'LN_'+Pos_Case+'.nii'
        T2_roiMaskPath = Pos_Path+'/'+Pos_Case+'/'+'2roi_'+Pos_Case+'.nii'
        T2_lnMaskPath = Pos_Path+'/'+Pos_Case+'/'+'2LN_'+Pos_Case+'.nii'
        
        T1_Img = sitk.ReadImage(T1_ImgPath)
        T1_roiMask = sitk.ReadImage(T1_roiMaskPath)
        
        T2_Img = sitk.ReadImage(T2_ImgPath)
        T2_roiMask = sitk.ReadImage(T2_roiMaskPath)
        T1_roifeatures, T1_roifeature_info = Extract_Features(T1_Img, T1_roiMask, 'params_T1.yaml')
        T2_roifeatures, T2_roifeature_info = Extract_Features(T2_Img, T2_roiMask, 'params_T2.yaml')
        T1_roifeatures['Name'] = Pos_Case
        T1_roifeatures['Class'] = 1
        T1_ROIFeature.append(T1_roifeatures)
        T2_roifeatures['Name'] = Pos_Case
        T2_roifeatures['Class'] = 1
        T2_ROIFeature.append(T2_roifeatures)
        try:
            T2_lnMask = sitk.ReadImage(T2_lnMaskPath)
            T1_lnMask = sitk.ReadImage(T1_lnMaskPath)
            T1_lnfeatures, T1_lnfeature_info = Extract_Features(T1_Img, T1_lnMask, 'params_T1.yaml')
            T2_lnfeatures, T2_lnfeature_info = Extract_Features(T2_Img, T2_lnMask, 'params_T2.yaml')
                    
    
            T1_lnfeatures['Name'] = Pos_Case
            T1_lnfeatures['Class'] = 1
            T1_LNFeature.append(T1_lnfeatures)
            T2_lnfeatures['Name'] = Pos_Case
            T2_lnfeatures['Class'] = 1
            T2_LNFeature.append(T2_lnfeatures)
        except:
            print(Pos_Case)
        
    for Neg_Case in tqdm(Neg_Lists):
        sleep(0.01)
        T1_ImgPath = Neg_Path+'/'+Neg_Case+'/'+'T1_'+Neg_Case+'.nii'
        T2_ImgPath = Neg_Path+'/'+Neg_Case+'/'+'T2_'+Neg_Case+'.nii'
        T1_roiMaskPath = Neg_Path+'/'+Neg_Case+'/'+'roi_'+Neg_Case+'.nii'
        T1_lnMaskPath = Neg_Path+'/'+Neg_Case+'/'+'LN_'+Neg_Case+'.nii'
        T2_roiMaskPath = Neg_Path+'/'+Neg_Case+'/'+'2roi_'+Neg_Case+'.nii'
        T2_lnMaskPath = Neg_Path+'/'+Neg_Case+'/'+'2LN_'+Neg_Case+'.nii'
        
        T1_Img = sitk.ReadImage(T1_ImgPath)
        T1_roiMask = sitk.ReadImage(T1_roiMaskPath)
        T1_lnMask = sitk.ReadImage(T1_lnMaskPath)
        T2_Img = sitk.ReadImage(T2_ImgPath)
        T2_roiMask = sitk.ReadImage(T2_roiMaskPath)
        T2_lnMask = sitk.ReadImage(T2_lnMaskPath)
        
        T1_roifeatures, T1_roifeature_info = Extract_Features(T1_Img, T1_roiMask, 'params_T1.yaml')
        T2_roifeatures, T2_roifeature_info = Extract_Features(T2_Img, T2_roiMask, 'params_T2.yaml')
        T1_lnfeatures, T1_lnfeature_info = Extract_Features(T1_Img, T1_lnMask, 'params_T1.yaml')
        T2_lnfeatures, T2_lnfeature_info = Extract_Features(T2_Img, T2_lnMask, 'params_T2.yaml')
                
        T1_roifeatures['Name'] = Neg_Case
        T1_roifeatures['Class'] = 0
        T1_ROIFeature.append(T1_roifeatures)
        T2_roifeatures['Name'] = Neg_Case
        T2_roifeatures['Class'] = 0
        T2_ROIFeature.append(T2_roifeatures)
        T1_lnfeatures['Name'] = Neg_Case
        T1_lnfeatures['Class'] = 0
        T1_LNFeature.append(T1_lnfeatures)
        T2_lnfeatures['Name'] = Neg_Case
        T2_lnfeatures['Class'] = 0
        T2_LNFeature.append(T2_lnfeatures)
    
    df = DF(T2_ROIFeature).fillna('0')
    df.to_csv('./T2_ROIFeature.csv',index=False,sep=',')
    df = DF(T1_ROIFeature).fillna('0')
    df.to_csv('./T1_ROIFeature.csv',index=False,sep=',')
    df = DF(T2_LNFeature).fillna('0')
    df.to_csv('./T2_LNFeature.csv',index=False,sep=',')
    df = DF(T1_LNFeature).fillna('0')
    df.to_csv('./T1_LNFeature.csv',index=False,sep=',')
    end = time.clock()
    print(end-start)  