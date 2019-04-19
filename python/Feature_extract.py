import math
import os
import pickle
import subprocess
from random import shuffle

import numpy as np
import numpy.matlib
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def handle_data(dic):
    morpho = []
    textu = []

    morpho.append(dic['volume'][0][0])
    morpho.append(dic['solidity'][0][0])
    morpho.append(dic['eccentricity'][0][0])
    morpho.append(dic['sizeROI'][0][0])

    textu = textu + cleanify(dic['textures_GLRLM'])
    textu = textu + cleanify(dic['textures_GLSZM'])
    textu = textu + cleanify(dic['textures_GLCM'])
    textu = textu + cleanify(dic['textures_NGTDM'])

    forder = cleanify(dic['textures_Global'])

    return morpho, textu, forder


def cleanify(utidy):
    tidy = []
    len_utidy = len(utidy[0][0])

    for i in range(len_utidy):
        tidy.append(utidy[0][0][i][0][0])

    return tidy


def get_bootsam(outcome, nBoot):
    ind_pos_temp = np.where(outcome == 1)[0]
    siz = len(ind_pos_temp) - 4
    ind_pos = np.random.choice(ind_pos_temp, size=siz, replace=False)
    ind_neg = np.where(outcome == 0)[0]
    nPos = len(ind_pos) + 4
    nNeg = len(ind_neg)
    nInst = nPos + nNeg
    ind_pos = np.matlib.repmat(ind_pos, nNeg, 1)
    ind_neg = np.matlib.repmat(ind_neg, nPos, 1)
    ind = list(ind_pos.flatten()) + list(ind_neg.flatten())
    nSize = len(ind)
    shuffle(ind)
    shuffle(ind)
    ind = np.array(ind)
    trainSet = ind[np.ceil((nSize * np.random.rand(nInst, nBoot)) - 1).astype('int')]

    testSet = {}
    for n in range(nBoot):
        vectTest = np.ones((nInst, 1))
        vectTest[trainSet[:, n]] = 0
        testSet[n] = np.where(vectTest == 1)[0]

    return trainSet, testSet


def applymine(var_Name, datafr, label):
    dataf = datafr.copy()
    dataf.loc[-1] = dataf.loc[var_Name]  # adding a row
    dataf.index = dataf.index + 1  # shifting index
    dataf.sort_index(inplace=True)
    csv_tmp = "./MINE/" + label + ".csv"
    dataf.to_csv(csv_tmp, header=False)
    out = subprocess.Popen(['java', '-jar', './MINE/MINE.jar', csv_tmp, '-masterVariable', '0'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    output, error = out.communicate()
    csv_name_res = "./MINE/" + label + ".csv,mv=0,cv=0.0,B=n^0.6,Results.csv"
    mic_df = pd.read_csv(csv_name_res)
    mic_df = mic_df.sort_values('Y var')
    return np.array(mic_df['MIC (strength)'])


patient_path = "./mat_files_for_extraction/"
load_path = "./mat_files_feat_extracted/"

Patients = os.listdir(patient_path)
nPatients = len(Patients)

data_dict = {}
i = 1
for patient in Patients:
    patient_name = patient.split('.')[0]
    data_dict[patient_name] = []
    file_path = os.path.join(load_path, patient)
    for Ng in [8, 16, 32, 64]:
        for voxel in range(1, 6):
            for norm in [1, 2]:
                file_name = patient + "_" + str(norm) + "_" + str(voxel) + "_" + str(Ng)
                print("Loading... " + file_name + " : " + str(i) + "/" + str(nPatients))
                file_dict = loadmat(file_path + "_" + str(norm) + "_" + str(voxel) + "_" + str(Ng))
                morpho, textu, forder = handle_data(file_dict)
                data_dict[patient_name] = data_dict[patient_name] + textu
    data_dict[patient_name] = data_dict[patient_name] + morpho + forder
    i += 1

df = pd.DataFrame(data_dict)
df = df.transpose()

var = []
for i in range(1607):
    var_name = "Variable" + str(i)
    var.append(var_name)

df.columns = var

col = ['Patient #', 'Sex', 'Age', 'Primary Site', 'T-stage', 'N-stage',
       'M-stage', 'TNM group stage', 'HPV status',
       'Time - diagnosis to diagnosis (days)',
       'Time - diagnosis to PET (days)', 'Time - diagnosis to CT sim (days)',
       'Time - diagnosis to start treatment (days)',
       'Time - diagnosis to end treatment (days)', 'Therapy', 'Surgery',
       'Time - diagnosis to last follow-up (days)', 'Locoregional',
       'Distant', 'Death', 'Time - diagnosis to LR (days)',
       'Time - diagnosis to DM (days)', 'Time - diagnosis to Death (days)']

xls_chum = pd.read_excel("./HeadNeck/INFOclinical_HN_Version2_30may2018.xlsx", sheet_name="CHUM",
                         skipfooter=6, names=col)
xls_chus = pd.read_excel("./HeadNeck/INFOclinical_HN_Version2_30may2018.xlsx", sheet_name="CHUS",
                         skipfooter=7, names=col)
xls_hgj = pd.read_excel("./HeadNeck/INFOclinical_HN_Version2_30may2018.xlsx", sheet_name="HGJ",
                        skipfooter=6, names=col)
xls_hmr = pd.read_excel("./HeadNeck/INFOclinical_HN_Version2_30may2018.xlsx", sheet_name="HMR",
                        skipfooter=5, names=col)
clini_df = pd.concat([xls_chum, xls_chus, xls_hgj, xls_hmr], sort=False)
clini_df.set_index("Patient #", inplace=True)

# Presence of different image resolutions
discarded_CT = clini_df.index.isin(["HN-CHUM-044", "HN-CHUM-045", "HN-CHUM-048", "HN-CHUM-059"])

total_CT = os.listdir("./HeadNeck/CT/")

for label in ['Distant', 'Locoregional', 'Death']:

    print("Label : " + label)
    y = clini_df.loc[total_CT, :][label]

    merged_df = df.merge(y, how='left', left_index=True, right_index=True, sort=True)

    chum_ind = merged_df.index[:58]
    chus_ind = merged_df.index[58:71]
    hgj_ind = merged_df.index[71:162]
    hmr_ind = merged_df.index[162:]

    train_df = merged_df.loc[list(chus_ind) + list(hgj_ind), :]
    test_df = merged_df.loc[list(chum_ind) + list(hmr_ind), :]

    X_train = train_df[train_df.columns[:-1]]
    y_train = train_df[label]
    X_test = test_df[test_df.columns[:-1]]
    y_test = test_df[label]

    X_train_text = X_train[X_train.columns[:-7]]

    applymine_df = X_train_text.copy()
    applymine_df.columns = range(1600)
    applymine_df = applymine_df.transpose()
    csv_name = "./MINE/" + label + ".csv"
    applymine_df.to_csv(csv_name, header=False)

    nBoot = 100
    trainS, testS = get_bootsam(y_train, nBoot)

    matcorr = np.zeros((X_train_text.shape[1], nBoot))
    for n in range(nBoot):
        dataBoot = X_train_text.iloc[sorted(list(trainS[:, n]))]
        outcomeBoot = y_train[sorted(list(trainS[:, n]))]
        matcorr[:, n] = abs(dataBoot.corrwith(outcomeBoot, axis=0, method='spearman'))
        print("Correlation Matrix, bootstrapping sample " + str(n))

    tin = pd.DataFrame(matcorr)
    coor = tin.sum(axis=1) / nBoot
    f1 = coor.values.argmax()

    nFeatures = 1600
    setSize = 25

    red_feat = []
    red_feat.append(f1)
    PICtest = np.zeros((nFeatures, setSize - 1))

    print("Feature Reduction")
    for f in range(setSize - 1):
        print("Feaure... " + str(f))
        varName = red_feat[f]
        print(str(varName) + " starting")
        for n in range(nBoot):
            print("Boot Sample... " + str(n))
            dataBoot = applymine_df.iloc[:, sorted(list(trainS[:, n]))]
            PICtest[:, f] = PICtest[:, f] + (1 - applymine(varName, dataBoot, label))
        print(str(varName) + " done")
        PICtest[:, f] = PICtest[:, f] / nBoot
        PICtemp = np.zeros((nFeatures, 1))
        for k in range(1, f + 2):
            fn = f + 1
            PICtemp = PICtemp + (float(2) * (fn - k + 1) / (fn * (fn + 1)) * PICtest[:, k - 1]).reshape(-1, 1)
        Gain = (0.5 * coor) + (0.5 * PICtemp[:, 0])
        best = 0
        sort_gain = Gain.sort_values(ascending=False)
        var_best = sort_gain.index[best]
        while (var_best in red_feat):
            best += 1
            var_best = sort_gain.index[best]
        red_feat.append(var_best)

    red_X_train = X_train_text.iloc[:, red_feat]
    red_X_test = X_test.iloc[:, red_feat]

    # feature selection
    print("Feature Selection")
    eps = math.ldexp(1.0, -1074)
    top = 1 - 1 / np.exp(1)
    low = 1 / np.exp(1)
    nFeat = red_X_train.shape[1]
    nInst = red_X_train.shape[0]

    maxOrder = 10
    modelMat = np.zeros((nFeat, maxOrder))
    metricMat = np.zeros((nFeat, maxOrder))

    for i in range(nFeat):
        indLeft = np.arange(nFeat)
        indLeft = np.delete(indLeft, i)
        modelMat[i, 0] = i
        X_train_lg1 = np.array(red_X_train.iloc[:, i]).reshape(-1, 1)
        mina = MinMaxScaler()
        mina.fit(X_train_lg1)
        trans_X_train = mina.transform(X_train_lg1)
        logistic = LogisticRegression()
        logistic.fit(trans_X_train, y_train)
        y_pred = logistic.predict(trans_X_train)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        sensData = tp / (tp + fn)
        specData = tn / (tn + fn)
        aucData = roc_auc_score(y_train, y_pred)
        aucTemp = 0
        sensTemp = 0
        specTemp = 0
        nBoot = 100
        for n in range(nBoot):
            Xtrain = np.array(red_X_train.iloc[trainS[:, n], i]).reshape(-1, 1)
            Xtest = np.array(red_X_train.iloc[testS[n], i]).reshape(-1, 1)
            Ytrain = y_train[trainS[:, n]]
            Ytest = y_train[testS[n]]
            mint = MinMaxScaler()
            mint.fit(Xtrain)
            trans_Xtrain = mint.transform(Xtrain)
            trans_Xtest = mint.transform(Xtest)
            logisticb = LogisticRegression()
            logisticb.fit(trans_Xtrain, Ytrain)
            y_pred = logisticb.predict(trans_Xtest)
            tn, fp, fn, tp = confusion_matrix(Ytest, y_pred).ravel()
            aucBoot = roc_auc_score(Ytest, y_pred)
            sensBoot = tp / (tp + fn)
            specBoot = tn / (tn + fn)
            alpha = top / (1 - low * (aucData - aucBoot) / (aucData - 0.5 + eps))
            if alpha > 1:
                alpha = 1
            elif alpha < top:
                alpha = top

            if aucBoot < 0.5:
                aucBoot = 0.5
            aucTemp = aucTemp + (1 - alpha) * aucData + alpha * aucBoot
            #           For SENSITIVITY
            alpha = top / (1 - low * (sensData - sensBoot) / (sensData + eps))
            if alpha < top:
                alpha = top
            sensTemp = sensTemp + (1 - alpha) * sensData + alpha * sensBoot
            #           For SPECIFICITY
            alpha = top / (1 - low * (specData - specBoot) / (specData + eps))
            if alpha < top:
                alpha = top
            specTemp = specTemp + (1 - alpha) * specData + alpha * specBoot

        aucTemp = aucTemp / nBoot
        sensTemp = sensTemp / nBoot
        specTemp = specTemp / nBoot
        metricMat[i, 0] = 0.5 * aucTemp + 0.5 * (1 - abs(sensTemp - specTemp))

        for j in range(1, maxOrder):
            maxMetric = 0
            for k in range(nFeat - j):
                print(str(i) + " " + str(j) + " " + str(k))
                indexModel = np.append(modelMat[i, 0:j], indLeft[k])
                Xtrain = red_X_train.iloc[:, indexModel]

                mina = MinMaxScaler()
                mina.fit(Xtrain)
                trans_Xtrain = mina.transform(Xtrain)

                logistic = LogisticRegression()
                logistic.fit(trans_Xtrain, y_train)
                y_pred = logistic.predict(trans_Xtrain)
                tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
                sensData = tp / (tp + fn)
                specData = tn / (tn + fn)
                aucData = roc_auc_score(y_train, y_pred)
                aucTemp = 0
                sensTemp = 0
                specTemp = 0
                nBoot = 100
                for n in range(nBoot):
                    Xtrain = red_X_train.iloc[trainS[:, n], indexModel]
                    Xtest = red_X_train.iloc[testS[n], indexModel]
                    Ytrain = y_train[trainS[:, n]]
                    Ytest = y_train[testS[n]]

                    mint = MinMaxScaler()
                    mint.fit(Xtrain)
                    trans_Xtrain = mint.transform(Xtrain)
                    trans_Xtest = mint.transform(Xtest)

                    logisticb = LogisticRegression()
                    logisticb.fit(trans_Xtrain, Ytrain)
                    y_pred = logisticb.predict(trans_Xtest)

                    tn, fp, fn, tp = confusion_matrix(Ytest, y_pred).ravel()
                    aucBoot = roc_auc_score(Ytest, y_pred)
                    sensBoot = tp / (tp + fn)
                    specBoot = tn / (tn + fn)

                    alpha = top / (1 - low * (aucData - aucBoot) / (aucData - 0.5 + eps))
                    if alpha > 1:
                        alpha = 1
                    elif alpha < top:
                        alpha = top

                    if aucBoot < 0.5:
                        aucBoot = 0.5
                    aucTemp = aucTemp + (1 - alpha) * aucData + alpha * aucBoot

                    #           For SENSITIVITY
                    alpha = top / (1 - low * (sensData - sensBoot) / (sensData + eps))
                    if alpha < top:
                        alpha = top
                    sensTemp = sensTemp + (1 - alpha) * sensData + alpha * sensBoot

                    #          % For SPECIFICITY
                    alpha = top / (1 - low * (specData - specBoot) / (specData + eps))
                    if alpha < top:
                        alpha = top
                    specTemp = specTemp + (1 - alpha) * specData + alpha * specBoot

                aucTemp = aucTemp / nBoot
                sensTemp = sensTemp / nBoot
                specTemp = specTemp / nBoot
                metricTemp = 0.5 * aucTemp + 0.5 * (1 - abs(sensTemp - specTemp))
                if metricTemp >= maxMetric:
                    maxMetric = metricTemp
                    index = indLeft[k]
            modelMat[i, j] = index
            metricMat[i, j] = maxMetric
            indLeft = np.delete(indLeft, np.where(indLeft == index))

    indMax = np.argmax(metricMat, axis=0)

    fname = "./" + label
    with open(fname, 'wb+') as fh:
        pickle.dump([X_train_text, X_test, y_train, y_test, red_feat, modelMat, metricMat], fh, protocol=2)
