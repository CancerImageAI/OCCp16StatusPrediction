# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:52:21 2022

@author: DELL
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from skfeature.function.similarity_based import reliefF
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel,mutual_info_classif
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score,matthews_corrcoef
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import scipy.stats as stats
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE
import xlrd
import rpy2.robjects as robj
r = robj.r
from rpy2.robjects.packages import importr

def roc_test_r(targets_1, scores_1, targets_2, scores_2, method='delong'):
    # method: “delong”, “bootstrap” or “venkatraman”
    importr('pROC')
    robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
    robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
    robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
    robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)

    r('roc_1 <- roc(targets_1, scores_1)')
    r('roc_2 <- roc(targets_2, scores_2)')
    # print(r('roc_1'),r('roc_2'))
    r('result = roc.test(roc_1, roc_2, method="%s")' % method)
    p_value = r('p_value = result$p.value')
    return np.array(p_value)[0]

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)
    F1_3 = f1_score(truth, predicted)
    print('F1:', F1_3)
    Precision = precision_score(truth, predicted)
    print('Precision:', Precision)
    
if __name__ == '__main__': 
    ##ROI feature
    T1_list_path = 'T1_ROIFeature.csv'
    f_T1 = open(T1_list_path)
    T1_Tumor_list = pd.read_csv(f_T1)
    List_Num = np.array(T1_Tumor_list['Name'].tolist())
    Class = np.array(T1_Tumor_list['Class'].tolist())

    T1_ROI_Feature = T1_Tumor_list.values[:,:-2]
    T1_ROI_Feature_Name = list(T1_Tumor_list.head(0))[:-2]
    T1_ROI_Feature_Name = ['T1-Tumor-'+i for i in T1_ROI_Feature_Name]
    
    T1_list_testpath = 'T1_ROIFeature_TestingData.csv'
    f_T1_test = open(T1_list_testpath)
    T1_Tumor_testlist = pd.read_csv(f_T1_test)
    List_Num_test = np.array(T1_Tumor_testlist['Name'].tolist())
    Class_Test = np.array(T1_Tumor_testlist['Class'].tolist())

    T1_ROI_Feature_Test = T1_Tumor_testlist.values[:,:-2]
    
    standard_scaler = StandardScaler()
    T1_ROI_Feature = standard_scaler.fit_transform(np.array(T1_ROI_Feature))
    T1_ROI_Feature_Test = standard_scaler.transform(np.array(T1_ROI_Feature_Test))
    # for i in range(3,11):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=5, step=1)
    
    T1_ROI_Feature_new = selector_Img.fit_transform(T1_ROI_Feature,Class)
    T1_ROI_Feature_new_Test = selector_Img.transform(T1_ROI_Feature_Test)
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T1_ROI_Feature_Name)[indices])
    T1_ROI_SelectedFeature_Name = np.array(T1_ROI_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(T1_ROI_Feature_new, Class)
    
    train_prob_T1_ROI = clf.predict_proba(T1_ROI_Feature_new)[:,1]
    pred_label_train = clf.predict(T1_ROI_Feature_new)
    fpr_tumor_T1_train,tpr_tumor_T1_train,threshold = roc_curve(Class, np.array(train_prob_T1_ROI)) ###计算真正率和假正率
    auc_score_tumor_T1_train = auc(fpr_tumor_T1_train,tpr_tumor_T1_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T1_ROI), Class)
    print("Training Dataset")
    print('T1 Tumor Feature AUC:%.2f'%auc_score_tumor_T1_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T1 Tumor Feature ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result = {}
    Train_Result['ID'] = List_Num
    Train_Result['Class'] = Class
    Train_Result['T1 Tumor Score'] = train_prob_T1_ROI
    
    test_prob_T1_ROI = clf.predict_proba(T1_ROI_Feature_new_Test)[:,1]
    pred_label = clf.predict(T1_ROI_Feature_new_Test)
    fpr_tumor_T1,tpr_tumor_T1,threshold = roc_curve(Class_Test, np.array(test_prob_T1_ROI)) ###计算真正率和假正率
    auc_score_tumor_T1 = auc(fpr_tumor_T1,tpr_tumor_T1)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T1_ROI), Class_Test)
    print("Testing Dataset")
    print('T1 Tumor Feature AUC:%.2f'%auc_score_tumor_T1,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T1 Tumor Feature ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label)*100))
    prediction_score(Class_Test,pred_label)
    print('-----------------------------------------')
    Test_Result = {}
    Test_Result['ID'] = List_Num_test
    Test_Result['Class'] = Class_Test
    Test_Result['T1 Tumor Score'] = test_prob_T1_ROI

    #LN Feature
    T1_LN_list_path = 'T1_LNFeature.csv'
    T1_f_LN = open(T1_LN_list_path)
    T1_LN_list = pd.read_csv(T1_f_LN)
    List_Num = np.array(T1_LN_list['Name'].tolist())
    Class = np.array(T1_LN_list['Class'].tolist())

    T1_LN_Feature = T1_LN_list.values[:,:-2]
    T1_LN_Feature_Name = list(T1_LN_list.head(0))[:-2]
    T1_LN_Feature_Name = ['T1-LN-'+i for i in T1_LN_Feature_Name]
    
    T1_LN_list_testpath = 'T1_LNFeature_TestingData.csv'
    T1_f_LN_test = open(T1_LN_list_testpath)
    T1_LN_testlist = pd.read_csv(T1_f_LN_test)
    List_Num_test = np.array(T1_LN_testlist['Name'].tolist())
    Class_Test = np.array(T1_LN_testlist['Class'].tolist())

    T1_LN_Feature_Test = T1_LN_testlist.values[:,:-2]

    standard_scaler = StandardScaler()
    T1_LN_Feature = standard_scaler.fit_transform(np.array(T1_LN_Feature))
    T1_LN_Feature_Test = standard_scaler.transform(np.array(T1_LN_Feature_Test))
    # for i in range(3,11):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=3, step=1)
    T1_LN_Feature_new = selector_Img.fit_transform(T1_LN_Feature,Class)
    T1_LN_Feature_new_Test = selector_Img.transform(T1_LN_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T1_LN_Feature_Name)[indices])
    T1_LN_SelectedFeature_Name = np.array(T1_LN_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(T1_LN_Feature_new, Class)
    
    train_prob_T1_LN = clf.predict_proba(T1_LN_Feature_new)[:,1]
    pred_label_train = clf.predict(T1_LN_Feature_new)
    fpr_LN_T1_train,tpr_LN_T1_train,threshold = roc_curve(Class, np.array(train_prob_T1_LN)) ###计算真正率和假正率
    auc_score_LN_T1_train = auc(fpr_LN_T1_train,tpr_LN_T1_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T1_LN), Class)
    print("Training Dataset")
    print('T1 LN Feature AUC:%.2f'%auc_score_LN_T1_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T1 LN Feature ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result['T1 LN Score'] = train_prob_T1_LN
    
    test_prob_T1_LN = clf.predict_proba(T1_LN_Feature_new_Test)[:,1]
    pred_label = clf.predict(T1_LN_Feature_new_Test)
    fpr_LN_T1,tpr_LN_T1,threshold = roc_curve(Class_Test, np.array(test_prob_T1_LN)) ###计算真正率和假正率
    auc_score_LN_T1 = auc(fpr_LN_T1,tpr_LN_T1)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T1_LN), Class_Test)
    print("Testing Dataset")
    print('T1 LN Feature AUC:%.2f'%auc_score_LN_T1,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T1 LN Feature ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label)*100))
    prediction_score(Class_Test,pred_label)
    print('-----------------------------------------')
    Test_Result['T1 LN Score'] = test_prob_T1_LN
    
    ##ROI feature
    T2_list_path = 'T2_ROIFeature.csv'
    f_T2 = open(T2_list_path)
    T2_Tumor_list = pd.read_csv(f_T2)
    List_Num = np.array(T2_Tumor_list['Name'].tolist())
    Class = np.array(T2_Tumor_list['Class'].tolist())

    T2_ROI_Feature = T2_Tumor_list.values[:,:-2]
    T2_ROI_Feature_Name = list(T2_Tumor_list.head(0))[:-2]
    T2_ROI_Feature_Name = ['T2-Tumor-'+i for i in T2_ROI_Feature_Name]
    
    T2_list_testpath = 'T2_ROIFeature_TestingData.csv'
    f_T2_test = open(T2_list_testpath)
    T2_Tumor_testlist = pd.read_csv(f_T2_test)
    List_Num_test = np.array(T2_Tumor_testlist['Name'].tolist())
    Class_Test = np.array(T2_Tumor_testlist['Class'].tolist())

    T2_ROI_Feature_Test = T2_Tumor_testlist.values[:,:-2]
    
    standard_scaler = StandardScaler()
    T2_ROI_Feature = standard_scaler.fit_transform(np.array(T2_ROI_Feature))
    T2_ROI_Feature_Test = standard_scaler.transform(np.array(T2_ROI_Feature_Test))
    
    # for i in range(3,12):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=3, step=1)
    
    T2_ROI_Feature_new = selector_Img.fit_transform(T2_ROI_Feature,Class)
    T2_ROI_Feature_new_Test = selector_Img.transform(T2_ROI_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T2_ROI_Feature_Name)[indices])
    T2_ROI_SelectedFeature_Name = np.array(T2_ROI_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="sigmoid", probability=True, random_state=42)
    clf.fit(T2_ROI_Feature_new, Class)
    
    train_prob_T2_ROI = clf.predict_proba(T2_ROI_Feature_new)[:,1]
    pred_label_train = clf.predict(T2_ROI_Feature_new)
    fpr_tumor_T2_train,tpr_tumor_T2_train,threshold = roc_curve(Class, np.array(train_prob_T2_ROI)) ###计算真正率和假正率
    auc_score_tumor_T2_train = auc(fpr_tumor_T2_train,tpr_tumor_T2_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T2_ROI), Class)
    print("Training Dataset")
    print('T2 Tumor Feature AUC:%.2f'%auc_score_tumor_T2_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T2 Tumor Feature ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result['T2 Tumor Score'] = train_prob_T2_ROI
    
    test_prob_T2_ROI = clf.predict_proba(T2_ROI_Feature_new_Test)[:,1]
    pred_label = clf.predict(T2_ROI_Feature_new_Test)
    fpr_tumor_T2,tpr_tumor_T2,threshold = roc_curve(Class_Test, np.array(test_prob_T2_ROI)) ###计算真正率和假正率
    auc_score_tumor_T2 = auc(fpr_tumor_T2,tpr_tumor_T2)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T2_ROI), Class_Test)
    print('Testing Dataset')
    print('T2 Tumor Feature AUC:%.2f'%auc_score_tumor_T2,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T2 Tumor Feature ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label)*100))
    prediction_score(Class_Test,pred_label)
    print('-----------------------------------------')
    Test_Result['T2 Tumor Score'] = test_prob_T2_ROI
    
    #LN Feature
    T2_LN_list_path = 'T2_LNFeature.csv'
    T2_f_LN = open(T2_LN_list_path)
    T2_LN_list = pd.read_csv(T2_f_LN)
    List_Num = np.array(T2_LN_list['Name'].tolist())
    Class = np.array(T2_LN_list['Class'].tolist())

    T2_LN_Feature = T2_LN_list.values[:,:-2]
    T2_LN_Feature_Name = list(T2_LN_list.head(0))[:-2]
    T2_LN_Feature_Name = ['T2-LN-'+i for i in T2_LN_Feature_Name]
    
    T2_LN_list_testpath = 'T2_LNFeature_TestingData.csv'
    T2_f_LN_test = open(T2_LN_list_testpath)
    T2_LN_testlist = pd.read_csv(T2_f_LN_test)
    List_Num_test = np.array(T2_LN_testlist['Name'].tolist())
    Class_Test = np.array(T2_LN_testlist['Class'].tolist())

    T2_LN_Feature_Test = T2_LN_testlist.values[:,:-2]
    
    standard_scaler = StandardScaler()
    T2_LN_Feature = standard_scaler.fit_transform(np.array(T2_LN_Feature))
    T2_LN_Feature_Test = standard_scaler.transform(np.array(T2_LN_Feature_Test))
    
    # for i in range(3,12):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=4, step=1)
   
    T2_LN_Feature_new = selector_Img.fit_transform(T2_LN_Feature,Class)
    T2_LN_Feature_new_Test = selector_Img.transform(T2_LN_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T2_LN_Feature_Name)[indices])
    T2_LN_SelectedFeature_Name = np.array(T2_LN_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(T2_LN_Feature_new, Class)
    
    train_prob_T2_LN = clf.predict_proba(T2_LN_Feature_new)[:,1]
    pred_label_train = clf.predict(T2_LN_Feature_new)
    fpr_LN_T2_train,tpr_LN_T2_train,threshold = roc_curve(Class, np.array(train_prob_T2_LN)) ###计算真正率和假正率
    auc_score_LN_T2_train = auc(fpr_LN_T2_train,tpr_LN_T2_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T2_LN), Class)
    print("Training Dataset")
    print('T2 LN Feature AUC:%.2f'%auc_score_LN_T2_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T2 LN Feature ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result['T2 LN Score'] = train_prob_T2_LN
    
    test_prob_T2_LN = clf.predict_proba(T2_LN_Feature_new_Test)[:,1]
    pred_label = clf.predict(T2_LN_Feature_new_Test)
    fpr_LN_T2,tpr_LN_T2,threshold = roc_curve(Class_Test, np.array(test_prob_T2_LN)) ###计算真正率和假正率
    auc_score_LN_T2 = auc(fpr_LN_T2,tpr_LN_T2)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T2_LN), Class_Test)
    print('Testing Dataset')
    print('T2 LN Feature AUC:%.2f'%auc_score_LN_T2,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T2 LN Feature ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label)*100))
    prediction_score(Class_Test,pred_label)
    print('-----------------------------------------')
    Test_Result['T2 LN Score'] = test_prob_T2_LN
    
    ##ROI Fusion 
    ROI_Fusion_Feature = np.hstack((T1_ROI_Feature,T2_ROI_Feature))
    ROI_Fusion_Feature_Test = np.hstack((T1_ROI_Feature_Test,T2_ROI_Feature_Test))
    ROI_Fusion_Feature_Name = T1_ROI_Feature_Name+T2_ROI_Feature_Name
    # for i in range(3,12):
    #     print('feature num:', i)
            
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=10, step=1)
    
    ROI_F_Feature_new = selector_Img.fit_transform(ROI_Fusion_Feature,Class)
    ROI_F_Feature_new_Test = selector_Img.transform(ROI_Fusion_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(ROI_Fusion_Feature_Name)[indices])
    ROI_Fusion_SelectedFeature_Name = np.array(ROI_Fusion_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(ROI_F_Feature_new, Class)
    
    train_prob_ROI = clf.predict_proba(ROI_F_Feature_new)[:,1]
    pred_label_train = clf.predict(ROI_F_Feature_new)
    fpr_roi_train,tpr_roi_train,threshold = roc_curve(Class, np.array(train_prob_ROI)) ###计算真正率和假正率
    auc_score_roi_train = auc(fpr_roi_train,tpr_roi_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_ROI), Class)
    print("Training Dataset")
    print('ROI Feature Fusion AUC:%.2f'%auc_score_roi_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('ROI Feature Fusion ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result['ROI Fusion Score'] = train_prob_ROI    
    
    test_prob_ROI = clf.predict_proba(ROI_F_Feature_new_Test)[:,1] 
    pred_label_ROI = clf.predict(ROI_F_Feature_new_Test)
    fpr_roi,tpr_roi,threshold = roc_curve(Class_Test, np.array(test_prob_ROI)) ###计算真正率和假正率
    auc_score_roi = auc(fpr_roi,tpr_roi)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_ROI), Class_Test)
    print('Testing Dataset')
    print('ROI Feature Fusion AUC:%.2f'%auc_score_roi,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('ROI Feature Fusion ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label_ROI)*100))
    prediction_score(Class_Test,pred_label_ROI)
    print('-----------------------------------------')
    Test_Result['ROI Fusion Score'] = test_prob_ROI
    
    ##LN Fusion 
    LN_Fusion_Feature = np.hstack((T1_LN_Feature_new,T2_LN_Feature_new))
    LN_Fusion_Feature_Test = np.hstack((T1_LN_Feature_new_Test,T2_LN_Feature_new_Test))
    LN_Fusion_Feature_Name = np.hstack((T1_LN_SelectedFeature_Name,T2_LN_SelectedFeature_Name))
    # for i in range(3,12):
    #     print('feature num:', i)
            
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=7, step=1)
    
    LN_F_Feature_new = selector_Img.fit_transform(LN_Fusion_Feature,Class)
    LN_F_Feature_new_Test = selector_Img.transform(LN_Fusion_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(LN_Fusion_Feature_Name)[indices])
    LN_Fusion_SelectedFeature_Name = np.array(LN_Fusion_Feature_Name)[indices]
    
    clf = svm.SVC(kernel="sigmoid", probability=True, random_state=42)
    clf.fit(LN_F_Feature_new, Class)
    
    train_prob_LN = clf.predict_proba(LN_F_Feature_new)[:,1]
    pred_label_train = clf.predict(LN_F_Feature_new)
    fpr_LN_train,tpr_LN_train,threshold = roc_curve(Class, np.array(train_prob_LN)) ###计算真正率和假正率
    auc_score_LN_train = auc(fpr_LN_train,tpr_LN_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_LN), Class)
    print("Training Dataset")
    print('LN Feature Fusion AUC:%.2f'%auc_score_LN_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('LN Feature Fusion ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)
    Train_Result['LN Fusion Score'] = train_prob_LN
    
    test_prob_LN = clf.predict_proba(LN_F_Feature_new_Test)[:,1] 
    pred_label_LN = clf.predict(LN_F_Feature_new_Test)
    fpr_LN,tpr_LN,threshold = roc_curve(Class_Test, np.array(test_prob_LN)) ###计算真正率和假正率
    auc_score_LN = auc(fpr_LN,tpr_LN)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_LN), Class_Test)
    print('Testing Dataset')
    print('LN Feature Fusion AUC:%.2f'%auc_score_LN,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('LN Feature Fusion ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label_LN)*100))
    prediction_score(Class_Test,pred_label_LN)  
    print('-----------------------------------------')
    Test_Result['LN Fusion Score'] = test_prob_LN
       
    ## T1 image fusion feature
    T1_Fusion_Feature = np.hstack((T1_ROI_Feature_new,T1_LN_Feature_new))
    T1_Fusion_Feature_Test = np.hstack((T1_ROI_Feature_new_Test,T1_LN_Feature_new_Test))
    T1_Fusion_Feature_Name = np.hstack((T1_ROI_SelectedFeature_Name,T1_LN_SelectedFeature_Name))
    # for i in range(3,12):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=4, step=1)
    
    F_Feature_new = selector_Img.fit_transform(T1_Fusion_Feature,Class)
    F_Feature_new_Test = selector_Img.transform(T1_Fusion_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T1_Fusion_Feature_Name)[indices])
    
    clf = svm.SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(F_Feature_new, Class)
    
    train_prob_T1 = clf.predict_proba(F_Feature_new)[:,1]
    pred_label_train = clf.predict(F_Feature_new)
    fpr_T1_train,tpr_T1_train,threshold = roc_curve(Class, np.array(train_prob_T1)) ###计算真正率和假正率
    auc_score_T1_train = auc(fpr_T1_train,tpr_T1_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T1), Class)
    print("Training Dataset")
    print('T1 Feature Fusion AUC:%.2f'%auc_score_T1_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T1 Feature Fusion ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train)  
    Train_Result['T1 Fusion Score'] = train_prob_T1
      
    test_prob_T1 = clf.predict_proba(F_Feature_new_Test)[:,1] 
    pred_label_T1 = clf.predict(F_Feature_new_Test)
    fpr_T1,tpr_T1,threshold = roc_curve(Class_Test, np.array(test_prob_T1)) ###计算真正率和假正率
    auc_score_T1 = auc(fpr_T1,tpr_T1)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T1), Class_Test)
    print('Testing Dataset')
    print('T1 Feature Fusion AUC:%.2f'%auc_score_T1,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T1 Feature Fusion ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label_T1)*100))
    prediction_score(Class_Test,pred_label_T1)  
    print('-----------------------------------------')
    Test_Result['T1 Fusion Score'] = test_prob_T1
    
    print('DeLong Test')
    print('-------------------------------------------------------')  
    ## T1 Model
    print('T1 Image Radiomics:')
    print('Fusion VS Tumor P-Value:',roc_test_r(Class_Test,test_prob_T1, Class_Test,test_prob_T1_ROI))
    ## Tumor vs ALL
    print('Fusion VS LN P-Value:',roc_test_r(Class_Test,test_prob_T1, Class_Test,test_prob_T1_LN))
    ## LN vs ALL
    print('Tumor VS LN P-Value:',roc_test_r(Class_Test,test_prob_T1_ROI, Class_Test,test_prob_T1_LN))
    print('-------------------------------------------------------') 
    
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)

    lw = 1.5
    plt.figure(figsize=(5,5))
    # fpr4,tpr4,threshold4 = roc_curve(np.array(real_class_Tumor),Fusion_max)
    plt.plot(fpr_T1, tpr_T1, color='red',
              lw=lw, label='Tumor+LN Feature (AUC=%.2f)'%(auc_score_T1))
    
    plt.plot(fpr_tumor_T1,tpr_tumor_T1, color='blue',
              lw=lw, label='Tumor Feature (AUC=%.2f)'%auc_score_tumor_T1) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_LN_T1,tpr_LN_T1, color='g',
              lw=lw, label='LN Feature (AUC=%.2f)'%auc_score_LN_T1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",edgecolor='k',title='T1 Image based Model',fontsize=10,fancybox=False)
    ax=plt.gca()
    
    ## T2 image fusion feature
    T2_Fusion_Feature = np.hstack((T2_ROI_Feature_new,T2_LN_Feature_new))
    T2_Fusion_Feature_Test = np.hstack((T2_ROI_Feature_new_Test,T2_LN_Feature_new_Test))
    T2_Fusion_Feature_Name = np.hstack((T2_ROI_SelectedFeature_Name,T2_LN_SelectedFeature_Name))
    # for i in range(3,12):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=7, step=1)
    
    F_Feature_new = selector_Img.fit_transform(T2_Fusion_Feature,Class)
    F_Feature_new_Test = selector_Img.transform(T2_Fusion_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(T2_Fusion_Feature_Name)[indices])
    
    clf = svm.SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(F_Feature_new, Class)
    
    train_prob_T2 = clf.predict_proba(F_Feature_new)[:,1]
    pred_label_train = clf.predict(F_Feature_new)
    fpr_T2_train,tpr_T2_train,threshold = roc_curve(Class, np.array(train_prob_T2)) ###计算真正率和假正率
    auc_score_T2_train = auc(fpr_T2_train,tpr_T2_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob_T2), Class)
    print("Training Dataset")
    print('T2 Feature Fusion AUC:%.2f'%auc_score_T2_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('T2 Feature Fusion ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train) 
    Train_Result['T2 Fusion Score'] = train_prob_T2
    
    test_prob_T2 = clf.predict_proba(F_Feature_new_Test)[:,1] 
    pred_label_T2 = clf.predict(F_Feature_new_Test)
    fpr_T2,tpr_T2,threshold = roc_curve(Class_Test, np.array(test_prob_T2)) ###计算真正率和假正率
    auc_score_T2 = auc(fpr_T2,tpr_T2)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob_T2), Class_Test)
    print("Testing Dataset")
    print('T2 Feature Fusion AUC:%.2f'%auc_score_T2,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('T2 Feature Fusion ACC:%.2f%%'%(accuracy_score(Class_Test,pred_label_T2)*100))  
    prediction_score(Class_Test,pred_label_T2) 
    Test_Result['T2 Fusion Score'] = test_prob_T2
    
    print('DeLong Test')
    print('-------------------------------------------------------')  
    ## T2 Model
    print('T2 Image Radiomics:')
    print('Fusion VS Tumor P-Value:',roc_test_r(Class_Test,test_prob_T2, Class_Test,test_prob_T2_ROI))
    ## Tumor vs ALL
    print('Fusion VS LN P-Value:',roc_test_r(Class_Test,test_prob_T2, Class_Test,test_prob_T2_LN))
    ## LN vs ALL
    print('Tumor VS LN P-Value:',roc_test_r(Class_Test,test_prob_T2_ROI, Class_Test,test_prob_T2_LN))
    print('-------------------------------------------------------') 
    
    plt.figure(figsize=(5,5))
    # fpr4,tpr4,threshold4 = roc_curve(np.array(real_class_Tumor),Fusion_max)
    plt.plot(fpr_T2, tpr_T2, color='red',
              lw=lw, label='Tumor+LN Feature (AUC=%.2f)'%(auc_score_T2))
    
    plt.plot(fpr_tumor_T2,tpr_tumor_T2, color='blue',
              lw=lw, label='Tumor Feature (AUC=%.2f)'%auc_score_tumor_T2) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_LN_T2,tpr_LN_T2, color='g',
              lw=lw, label='LN Feature (AUC=%.2f)'%auc_score_LN_T2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",edgecolor='k',title='T2 Image based Model',fontsize=10,fancybox=False)
    ax=plt.gca()
    
    ##ALL Fusion 
    Fusion_Feature = np.hstack((ROI_F_Feature_new,LN_F_Feature_new))
    Fusion_Feature_Test = np.hstack((ROI_F_Feature_new_Test,LN_F_Feature_new_Test))
    Fusion_Feature_Name = np.hstack((ROI_Fusion_SelectedFeature_Name, LN_Fusion_SelectedFeature_Name))
    # for i in range(3,12):
    #     print('feature num:', i)
    estimator = SVC(kernel="linear")
    selector_Img = RFE(estimator, n_features_to_select=6, step=1)
          
    F_Feature_new = selector_Img.fit_transform(Fusion_Feature,Class)
    F_Feature_new_Test = selector_Img.transform(Fusion_Feature_Test)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    print(np.array(Fusion_Feature_Name)[indices])
    
    clf = svm.SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(F_Feature_new, Class)
    
    train_prob = clf.predict_proba(F_Feature_new)[:,1]
    pred_label_train = clf.predict(F_Feature_new)
    fpr_train,tpr_train,threshold = roc_curve(Class, np.array(train_prob)) ###计算真正率和假正率
    auc_score_train = auc(fpr_train,tpr_train)
    auc_l_train, auc_h_train, auc_std_train = confindence_interval_compute(np.array(train_prob), Class)
    print("Training Dataset")
    print('All Feature Fusion AUC:%.2f'%auc_score_train,'+/-%.2f'%auc_std_train,'  95% CI:[','%.2f,'%auc_l_train,'%.2f'%auc_h_train,']')
    print('All Feature Fusion ACC:%.2f%%'%(accuracy_score(Class,pred_label_train)*100))
    prediction_score(Class,pred_label_train) 
    Train_Result['All Fusion Score'] = train_prob
    df = DF(Train_Result).fillna('0')
    df.to_csv('../Results/Train_Result.csv', index = False, sep=',')
    
    test_prob = clf.predict_proba(F_Feature_new_Test)[:,1] 
    pred_label = clf.predict(F_Feature_new_Test)
    fpr,tpr,threshold = roc_curve(Class_Test, np.array(test_prob)) ###计算真正率和假正率
    auc_score = auc(fpr,tpr)
    auc_l, auc_h, auc_std = confindence_interval_compute(np.array(test_prob), Class_Test)
    print('Testing Dataset')
    print('All Feature Fusion AUC:%.2f'%auc_score,'+/-%.2f'%auc_std,'  95% CI:[','%.2f,'%auc_l,'%.2f'%auc_h,']')
    print('All Feature Fusion ACC:%.2f%%'%(accuracy_score(Class_Test,test_prob>0.4)*100))
    prediction_score(Class_Test,test_prob>0.4) 
    Test_Result['All Fusion Score'] = test_prob
    df = DF(Test_Result).fillna('0')
    df.to_csv('../Results/Test_Result.csv', index = False, sep=',')
    
    print('DeLong Test')
    print('-------------------------------------------------------')  
    ## All Feature Model
    print('All Image Radiomics:')
    print('Fusion VS T1 P-Value:',roc_test_r(Class_Test,test_prob, Class_Test,test_prob_T1))
    ## T1 vs ALL
    print('Fusion VS T2 P-Value:',roc_test_r(Class_Test,test_prob, Class_Test,test_prob_T2))
    ## T2 vs ALL
    print('T1 VS T2 P-Value:',roc_test_r(Class_Test,test_prob_T1, Class_Test,test_prob_T2))
    
    print('Fusion VS ROI P-Value:',roc_test_r(Class_Test,test_prob, Class_Test,test_prob_ROI))
    ## T1 vs ALL
    print('Fusion VS LN P-Value:',roc_test_r(Class_Test,test_prob, Class_Test,test_prob_LN))
    ## T2 vs ALL
    print('ROI VS LN P-Value:',roc_test_r(Class_Test,test_prob_ROI, Class_Test,test_prob_LN))
    print('-------------------------------------------------------') 
    
    lw = 1.5
    plt.figure(figsize=(5,5))
    # fpr4,tpr4,threshold4 = roc_curve(np.array(real_class_Tumor),Fusion_max)
    plt.plot(fpr, tpr, color='red',
              lw=lw, label='Overall Image Feature (AUC=%.2f)'%auc_score)
    
    plt.plot(fpr_T1, tpr_T1, color='blueviolet',
            lw=lw, label='All T1 Image Feature (AUC=%.2f)'%auc_score_T1)
    
    plt.plot(fpr_T2, tpr_T2, color='darkorange',
            lw=lw, label='All T2 Image Feature (AUC=%.2f)'%auc_score_T2)
    
    plt.plot(fpr_roi,tpr_roi, color='blue',
              lw=lw, label='All Tumor Feature (AUC=%.2f)'%auc_score_roi) ###'ROC curve (area = %0.3f)' % auc假正率为横坐标，真正率为纵坐标做曲线fusion_auc

    plt.plot(fpr_LN,tpr_LN, color='g',
              lw=lw, label='All LN Feature (AUC=%.2f)'%auc_score_LN)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",edgecolor='k',title='Fusion of T1 and T2 Image Feature',fontsize=10,fancybox=False)
    ax=plt.gca()