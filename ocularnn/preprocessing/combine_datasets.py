#!/usr/bin/env python3
# coding: utf-8

import os
import random
import sys
import datetime

import tensorflow as tf
import tqdm
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (multilabel_confusion_matrix, ConfusionMatrixDisplay,
        classification_report, roc_auc_score, roc_curve, f1_score)

# Setting keras backend alias
K = keras.backend

# The labels are per patient, however I can use the diagnostic keywords
# to create labels for each individual eye
left_keywords = labels.iloc[:, 5].unique()
right_keywords = labels.iloc[:, 6].unique()
keywords = np.concatenate([left_keywords, right_keywords])

def split_diagnoses(diagnosis):
    """
    Split up the diagnosis strings into a list. Annoyingly there are
    two different types of commas delineating the diagnostic labels
    
    Parameters
    ----------
    diagnosis: set
         Contains all the diagnostic labels in the dataset.
         
    Returns
    -------
    words: list
        The split diagnostic labels.
    """
    if '，' in diagnosis:
        words = diagnosis.split('，')
    else:
        words = diagnosis.split(',')
    return words 

def get_diagnoses(diagnoses):
    diagnosis_set = set()
    for diagnosis in diagnoses:
        # There are annoyingly two different commas separating the diagnoses
        words = split_diagnoses(diagnosis)
        for j in words:
            diagnosis_set.add(j)
    return diagnosis_set

# Search for a given substring within diagnoses in the set
def get_kw(keyword, diagnoses_set):
    keyword_set = set()
    for kw in diagnoses_set:
        if keyword in kw:
            keyword_set.add(kw)
    return keyword_set

def create_kw(diagnoses_set):
    # Generate mappings for each disease
    normal_kw = set(['normal fundus'])
    non_indicator_kw = set(['lens dust', 'optic disk photographically invisible', 'anterior segment image', 'no fundus image', 'image offset', 'low image quality'])
    retinopathy_kw = get_kw('retinopathy', diagnoses_set)
    glaucoma_kw = get_kw('glaucoma', diagnoses_set)
    cataract_kw = get_kw('cataract', diagnoses_set)
    amd_kw = get_kw('age-related', diagnoses_set)
    hypertension_kw = get_kw('hypertens', diagnoses_set)
    myopia_kw = get_kw('myop', diagnoses_set)

    # Removing instances that belong to another class
    retinopathy_kw = retinopathy_kw.difference(myopia_kw)
    retinopathy_kw.remove('central serous chorioretinopathy')
    retinopathy_kw.remove('old chorioretinopathy')
    retinopathy_kw.remove('hypertensive retinopathy')

    # Determining the diagnoses present in the "other" category
    named_labels = normal_kw.union(retinopathy_kw, myopia_kw, glaucoma_kw, hypertension_kw, amd_kw, cataract_kw)
    other_kw = diagnoses.difference(named_labels.union(non_indicator_kw))
    
    kw = [normal_kw, retinopathy_kw, glaucoma_kw, cataract_kw, amd_kw, hypertension_kw, myopia_kw, other_kw]
    return kw

def match_diagnoses(diagnosis, df):
    """Some testing code to ensure that my algorithm will properly
    label the left and right fundus images using the diagnostic labels"""
    left = df.iloc[:, 5]
    left_index = left.index
    right = df.iloc[:, 6]
    right_index = right.index

    left_valid = set()
    right_valid = set()
    for ind, (left_d, right_d) in enumerate(zip(left.tolist(), right.tolist())):
        for d in diagnosis:
            if d in left_d:
                left_valid.add(left_index[ind])
            if d in right_d:
                 right_valid.add(right_index[ind])
                
    indices = list(left_valid.union(right_valid))
    return df.iloc[indices]

all_kw = create_kw(diagnoses)
cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

for kw, label in zip(all_kw, cols):
    test = match_diagnoses(kw, labels)
    print(test[label].sum(), labels[label].sum())


def label_odir(keyword_sets, df):
    left = df.iloc[:, 5]
    left_index = left.index
    right = df.iloc[:, 6]
    right_index = right.index
    
    patient_id = []
    img_names = []
    labels = []
    diagnoses = []
    
    for dfind, (left_d, right_d) in enumerate(zip(left.tolist(), right.tolist())):
        l_one_hot = np.zeros(8, dtype=int)
        r_one_hot = np.zeros(8, dtype=int)
        ldiag = split_diagnoses(left_d)
        rdiag = split_diagnoses(right_d)
        for d in ldiag:
            for ind, kwset in enumerate(keyword_sets):
                if d in kwset:
                    l_one_hot[ind] = 1
                    break
        for d in rdiag:
            for ind, kwset in enumerate(keyword_sets):
                if d in kwset:
                    r_one_hot[ind] = 1
                    break
        row = df.iloc[dfind]
        patient_id.extend([row['ID']] * 2)
        img_names.extend([row['Left-Fundus'], row['Right-Fundus']])
        diagnoses.append(','.join(ldiag))
        diagnoses.append(','.join(rdiag))
        labels.append(l_one_hot)
        labels.append(r_one_hot)
        
    columns = ['ID', 'image_name', 'diagnosis', 'N', 'DR', 'G', 'C', 'AMD', 'H', 'M', 'O']
    patient_id = np.asarray(patient_id).reshape(-1, 1)
    img_names = np.asarray(img_names).reshape(-1, 1)
    diagnoses = np.asarray(diagnoses).reshape(-1, 1)
    labels = np.asarray(labels)
    dataset = np.concatenate([patient_id, img_names, diagnoses, labels], axis=1)
    dataset = pd.DataFrame(dataset, columns=columns)
    return dataset

# Drop rows where there are no labels, which means the fundus images lack
# any labels with actionable indicators of disease (labels in the non_indicator_kw set)
odir_labeled = label_odir(all_kw, labels)
no_indicators = odir_labeled[odir_labeled.iloc[:, 3:].sum(axis=1) == 0].index
odir_labeled = odir_labeled.drop(no_indicators)
ODIR_LABELS = './../data/odir/odir_train.csv'
odir_labeled.to_csv(ODIR_LABELS, index=False)

# 30 rows dropped in total
odir_labeled.tail()


# Create one single .csv containing the categories, image names
# as well as the dataset where each image originated from.
g1020 = pd.read_csv(G1020_LABELS)
idrid = pd.read_csv(IDRID_LABELS)
idrid_test = pd.read_csv(IDRID_TEST)
refuge = pd.read_csv(REFUGE_LABELS)
refuge_valid = pd.read_excel(REFUGE_VALID)
adam = pd.read_csv(ADAM_LABELS)
messidor = pd.read_csv(MESSIDOR_LABELS)
palm = pd.read_csv(PALM_LABELS)
rfmid = pd.read_csv(RFMID_LABELS)
odir = pd.read_csv(ODIR_LABELS)




# Converting IDRiD and Messidor-2 to binary labels
idrid_dr = idrid['Retinopathy grade']
idrid_dr = np.where(idrid_dr > 0, 1, 0).reshape(-1, 1)
idrid['Retinopathy grade'] = idrid_dr
idrid['Image name'] += '.jpg'

# IDRiD test
idrid_test_dr = idrid_test['Retinopathy grade']
idrid_test_dr = np.where(idrid_test_dr > 0, 1, 0).reshape(-1, 1)
idrid_test['Retinopathy grade'] = idrid_test_dr
idrid_test['Image name'] = 'T' + idrid_test['Image name'] + '.jpg'

# Messidor-2
messidor_dr = messidor['adjudicated_dr_grade']
messidor_dr = np.where(messidor_dr > 0, 1, 0).reshape(-1, 1)
messidor['adjudicated_dr_grade'] = messidor_dr




# Formatting all the datasets and combining them into one
def format_binary(image_col, label_col, category, origin, dset_cols, df):
    # Getting image and label columns
    image_names = df[image_col] 
    image_names = image_names.rename('image_name')
    labels = df[label_col]
    dset_cat = {'DR': 1, 'G': 2, 'C': 3, 'AMD': 4, 'H': 5, 'M': 6, 'O': 7}

    # Creating array of one hot vectors for disease positive
    disease_pos = labels[labels == 1]
    disease_ind = disease_pos.index
    disease_labels = disease_pos.to_numpy()
    pos_one_hots = np.zeros((disease_labels.shape[0], 8), dtype=object)
    pos_one_hots[:, dset_cat[category]] = disease_pos

    # Creating array of one hot vectors for healthy
    healthy = labels[labels == 0]
    healthy_ind = healthy.index
    healthy_labels = np.ones(healthy.to_numpy().shape, dtype=int)
    neg_one_hots = np.zeros((healthy_labels.shape[0], 8), dtype=object)
    neg_one_hots[:, 0] = healthy_labels

    # Concatenating the images with the labels
    disease_pos = np.concatenate([image_names[disease_ind].to_numpy().reshape(-1, 1),
                    np.array([origin] * pos_one_hots.shape[0]).reshape(-1, 1),
                    pos_one_hots], axis=1)

    healthy = np.concatenate([image_names[healthy_ind].to_numpy().reshape(-1, 1),
                    np.array([origin] * neg_one_hots.shape[0]).reshape(-1, 1),
                    neg_one_hots], axis=1)

    df_formatted = pd.DataFrame(np.concatenate([disease_pos, healthy], axis=0), columns=dset_cols)
    return df_formatted

# We will use the same categories as ODIR for the dataset and also add an origin 
# column to better identify which datasets each image belongs to
dset = odir.drop(['ID', 'diagnosis'], axis=1)
dset.insert(1, 'origin', ['ODIR'] * odir.shape[0])

# Concatenating all the datasets with binary labels, only one that isn't is RFMiD
dset = dset.append(format_binary('imageID', 'binaryLabels', 'G', 'G1020', dset.columns, g1020))
dset = dset.append(format_binary('Image name', 'Retinopathy grade', 'DR', 'IDRiD', dset.columns, idrid))
dset = dset.append(format_binary('Image name', 'Retinopathy grade', 'DR', 'IDRiD', dset.columns, idrid_test))
dset = dset.append(format_binary('ImgName', 'glaucoma', 'G', 'REFUGE', dset.columns, refuge))
dset = dset.append(format_binary('ImgName', 'Glaucoma Label', 'G', 'REFUGE', dset.columns, refuge_valid))
dset = dset.append(format_binary('image_names', 'AMD', 'AMD', 'ADAM', dset.columns, adam))
dset = dset.append(format_binary('image_id', 'adjudicated_dr_grade', 'DR', 'MESSIDOR-2', dset.columns, messidor))
dset = dset.append(format_binary('image_names', 'myopia', 'M', 'PALM', dset.columns, palm))

# Formatting and appending RFMiD
rfmid_images = rfmid['ID'].astype('U') + '.png'
rfmid_images.rename('image_name')

# Diseased instances
rfmid_disease = rfmid[rfmid['Disease_Risk'] == 1]
rfmid_disease_ind = rfmid_disease.index
pos_one_hots = np.zeros((rfmid_disease.shape[0], 8), dtype=int)

# DR
rfmid_dr = rfmid_disease['DR'].astype(int)
pos_one_hots[:, 1] = rfmid_dr

# AMD
rfmid_amd = rfmid_disease['ARMD'].astype(int)
pos_one_hots[:, 4] = rfmid_amd

# Myopia
rfmid_myopia = rfmid_disease['MYA'].astype(int)
pos_one_hots[:, 6] = rfmid_myopia

# Other diseases
rfmid_other = (rfmid_disease.drop(['DR', 'ARMD', 'MYA'], axis=1).iloc[:, 2:].sum(axis=1) > 0).astype(int)
pos_one_hots[:, 7] = rfmid_other

# Healthy
rfmid_healthy = rfmid[rfmid['Disease_Risk'] == 0]
rfmid_healthy_ind = rfmid_healthy.index
neg_one_hots = np.zeros((rfmid_healthy.shape[0], 8), dtype=int)
neg_one_hots[:, 0] = np.ones(rfmid_healthy.shape[0], dtype=int)

# Concatenating the images with the labels
rfmid_disease = np.concatenate([rfmid_images[rfmid_disease_ind].to_numpy().reshape(-1, 1),
                np.array(['RFMiD'] * pos_one_hots.shape[0]).reshape(-1, 1),
                pos_one_hots], axis=1)

rfmid_healthy = np.concatenate([rfmid_images[rfmid_healthy_ind].to_numpy().reshape(-1, 1),
                np.array(['RFMiD'] * neg_one_hots.shape[0]).reshape(-1, 1),
                neg_one_hots], axis=1)

rfmid_formatted = pd.DataFrame(np.concatenate([rfmid_disease, rfmid_healthy], axis=0), columns=dset.columns)
dset = dset.append(rfmid_formatted)

# Resetting index
dset.index = range(dset.shape[0])
dset.tail()

# Saving labels to dataset folder where all images are in the same directory
dset.to_csv(os.path.join(os.curdir, '../dataset/dataset.csv'), index=False)


# Checking if all images are present
DATASET_IMAGES = os.path.join(os.curdir, '../dataset/images/')
dataset_dirnames = set(os.listdir(DATASET_IMAGES))
dataset_image_names = set(dset['image_name'].tolist())
dataset_image_names.difference(dataset_dirnames)
