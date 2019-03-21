import os
from dicom_contour.contour import *
import pydicom
import numpy as np
from scipy.io import savemat
import pandas as pd


PATH = "/home/aakif/HeadNeck/CT"


def coord(contour_dataset, path):

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    temp_files = os.listdir(path)
    for file in temp_files:
        file_path = os.path.join(path, file)
        img = pydicom.read_file(file_path)
        uid = img.SOPInstanceUID
        if uid == img_ID:
            break

    # img = dicom.read_file(path + img_ID + '.dcm')
    img_arr = img.pixel_array

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((y - origin_y) / y_spacing), np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                             shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, img_ID



def cfile2pixels(file, path, ROIContourSeq=0):

    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(path + file)
    # index 0 means that we are getting RTV information
    RTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in RTV.ContourSequence]
    img_contour_arrays = [coord(cdata, path) for cdata in contours]
    return img_contour_arrays

def get_data(path, index):

    images = []
    contours = []
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get contour file
    contour_file = "RTstructCT"
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    contour_dict = get_contour_dict(contour_file, path, index)

    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
        # get data from dicom.read_file
        else:
            fna = id_to_name[k]
            img_arr = dicom.read_file(path + fna).pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)


def create_image_mask_files(path, index):

    # Extract Arrays from DICOM
    X, Y = get_data(path, index)
    Y = np.array([fill_contour(y) if y.max() == 1 else y for y in Y])
    return X, Y

def get_contour_dict(contour_file, path, index):

    # handle `/` missing
    if path[-1] != '/': path += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path, index)

    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        contour_dict[img_id] = [img_arr, contour_arr]

    return contour_dict


xls_chum = pd.read_excel("./INFO_GTVcontours_HN.xlsx", sheet_name="CHUM")
chum = list(xls_chum.Patient)

xls_chus = pd.read_excel("./INFO_GTVcontours_HN.xlsx", sheet_name="CHUS")
chus = list(xls_chus.Patient)

xls_hgj = pd.read_excel("./INFO_GTVcontours_HN.xlsx", sheet_name="HGJ")
hgj = list(xls_hgj.Patient)

xls_hmr = pd.read_excel("./INFO_GTVcontours_HN.xlsx", sheet_name="HMR")
hmr = list(xls_hmr.Patient)

patients = os.listdir(PATH)
npatients = str(len(patients))
i=0
for patient in patients:
    i+=1
    print("Converting " + patient + " : " + str(i)+"/" + npatients)

    contour_path = os.path.join(os.path.join(PATH,patient),"RTstructCT")
    contour_file_mine = pydicom.read_file(contour_path)

    if patient in chum:
        gtv = xls_chum[xls_chum['Patient'] == patient]['Name GTV Primary']
        gtv = list(gtv)[0]
        gtv = gtv.upper()
        temp = get_roi_names(contour_file_mine)
        contour_list = [a.upper() for a in temp]
        contour_id = contour_list.index(gtv)

    elif patient in chus:
        gtv = xls_chus[xls_chus['Patient'] == patient]['Name GTV Primary']
        gtv = list(gtv)[0]
        gtv = gtv.upper()
        temp = get_roi_names(contour_file_mine)
        contour_list = [a.upper() for a in temp]
        contour_id = contour_list.index(gtv)

    elif patient in hgj:
        gtv = xls_hgj[xls_hgj['Patient'] == patient]['Name GTV Primary']
        gtv = list(gtv)[0]
        gtv = gtv.upper()
        temp = get_roi_names(contour_file_mine)
        contour_list = [a.upper() for a in temp]
        contour_id = contour_list.index(gtv)

    elif patient in hmr:
        gtv = xls_hmr[xls_hmr['Patient'] == patient]['Name GTV Primary']
        gtv = list(gtv)[0]
        gtv = gtv.upper()
        temp = get_roi_names(contour_file_mine)
        contour_list = [a.upper() for a in temp]
        contour_id = contour_list.index(gtv)
    else:
        print("ERROR in contour")
        break

    files_path = os.path.join(PATH,patient)
    files = os.listdir(files_path)

    id_to_name = {}
    for file in files:
        file_path = os.path.join(files_path, file)
        dcm_file = pydicom.read_file(file_path)
        id_to_name[dcm_file.SOPInstanceUID] = file

    slice_pixel_path = os.path.join(files_path,"000002.dcm")
    slice_pixel = pydicom.read_file(slice_pixel_path)
    sliceS = slice_pixel.SliceThickness
    pixelW = slice_pixel.PixelSpacing[0]
    img_data, mask = create_image_mask_files(files_path, contour_id)

    boundindex = np.where(mask == 1)
    boxBound = [[5, 5], [5, 5], [5, 5]]
    boxBound[0][0] = min(boundindex[0]) - 2
    boxBound[0][1] = max(boundindex[0]) + 2
    boxBound[1][0] = min(boundindex[1]) - 2
    boxBound[1][1] = max(boundindex[1]) + 2
    boxBound[2][0] = min(boundindex[2]) - 2
    boxBound[2][1] = max(boundindex[2]) + 2
    new_img_data = img_data[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    new_mask = mask[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    new_mask = new_mask.astype('float')
    # new_mask[new_mask == 0] = np.nan
    # roi = np.multiply(new_img_data, new_mask)
    savemat("/media/aakif/Common/MATLAB_files_both/" + patient + '.mat', mdict={'ROIbox': new_img_data,'mask':new_mask,
                                                                           'pixelW': pixelW, 'sliceS': sliceS})
