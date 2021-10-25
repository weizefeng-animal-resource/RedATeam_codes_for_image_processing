import cv2
import numpy as np
import os
from PIL import Image as Im


mat_ATeam_RedATeam = np.array([[699.0248, 100.322, 7.728962, 2.8675],
                               [21.06228, 1169.046, 76.75406, 16.99283],
                               [11.22265, 507.9494, 1048.082, 470.7074],
                               [68.21611, 23.51226, 312.2915, 631.3702]])
mat_ATeam_RedATeam = np.linalg.inv(mat_ATeam_RedATeam)
mat_Pyronic_RedATeam = np.array([[740.0754, 146.6344, 7.95356, 1.871],
                                 [23.37488, 1569.816, 99.76844, 22.68225],
                                 [11.22265, 507.9494, 1048.082, 470.7074],
                                 [68.21611, 23.51226, 312.2915, 631.3702]])
mat_Pyronic_RedATeam = np.linalg.inv(mat_Pyronic_RedATeam)


# edit it #########################
folder_name = ''  # the name of folder containing images
maximum_T = None
matrix = mat_ATeam_RedATeam  # , mat_Pyronic_RedATeam or else
subtract_background = True
subtract_background_size = None
#
do_median_blur = None
median_filter_size = None
###################################


def max_contrast_16bit(src):
    src = np.float32(src)
    min_intensity = src.min()
    if src.max() == 0:
        max_intensity = 1
    else:
        max_intensity = src.max()
    return np.int64((src - min_intensity) / (max_intensity - min_intensity) * 65535)


def max_contrast_8bit(src):
    src = np.float32(src)
    min_intensity = src.min()
    if src.max() == 0:
        max_intensity = 1
    else:
        max_intensity = src.max()
    return np.int64((src - min_intensity) / (max_intensity - min_intensity) * 255)


# mouse event to define the center of background region
def mouse_event_1(event, x, y, flags, param):
    global image_show, image_raw, x_temp, y_temp, center_of_background
    if event == cv2.EVENT_LBUTTONUP:
        if x == x_temp and y == y_temp:
            image_show = image_raw.copy()
            center_of_background = (x, y)
            cv2.circle(image_show, center_of_background, subtract_background_size, 255, 1)
        x_temp = x
        y_temp = y


# this program was designed to process the default raw data output of FV1000, Olympus, multi-channel time lapse mode
# for example, image of second channel at t=12 has a name: name_of_experiment_C002T012.tif
file_name_list = os.listdir(folder_name)
address_list = [['*', '*', '*', '*'] for _ in range(maximum_T)]
for file_name in file_name_list:
    if file_name[-4:] == '.tif':
        channel_number = int(file_name[-11:-8])
        time_number = int(file_name[-7:-4])
        address_list[time_number - 1][channel_number - 1] = file_name
if '*' in address_list:
    print('Error: images not found')
result_1 = []
result_2 = []
result_3 = []
result_4 = []
for i in range(maximum_T):
    print('processing images ... T = {0}'.format(i))
    image_ch1 = np.int64(cv2.imread(folder_name + '/' + address_list[i][0], -1))
    image_ch2 = np.int64(cv2.imread(folder_name + '/' + address_list[i][1], -1))
    image_ch3 = np.int64(cv2.imread(folder_name + '/' + address_list[i][2], -1))
    image_ch4 = np.int64(cv2.imread(folder_name + '/' + address_list[i][3], -1))
    #
    # get background region
    if subtract_background is True and i == 0:
        image_raw = np.uint8(max_contrast_8bit(image_ch1))
        image_show = image_raw.copy()
        x_temp = 0
        y_temp = 0
        center_of_background = tuple()
        cv2.namedWindow("acquiring background area", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("acquiring background area", mouse_event_1)
        while True:
            cv2.imshow("acquiring background area", image_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        mask = np.uint8(np.zeros(image_raw.shape))
        cv2.circle(mask, center_of_background, subtract_background_size, 1, -1)
        mask = np.int64(mask)
        mask_size = np.sum(mask)
    #
    if do_median_blur > 0:
        for _ in range(do_median_blur):
            image_ch1 = np.int64(cv2.medianBlur(np.float32(image_ch1), median_filter_size))
            image_ch2 = np.int64(cv2.medianBlur(np.float32(image_ch2), median_filter_size))
            image_ch3 = np.int64(cv2.medianBlur(np.float32(image_ch3), median_filter_size))
            image_ch4 = np.int64(cv2.medianBlur(np.float32(image_ch4), median_filter_size))
    #
    if subtract_background is True:
        background_ch1 = np.sum(image_ch1 * mask) / mask_size
        background_ch2 = np.sum(image_ch2 * mask) / mask_size
        background_ch3 = np.sum(image_ch3 * mask) / mask_size
        background_ch4 = np.sum(image_ch4 * mask) / mask_size
        image_ch1 = np.maximum(image_ch1 - background_ch1, 0)
        image_ch2 = np.maximum(image_ch2 - background_ch1, 0)
        image_ch3 = np.maximum(image_ch3 - background_ch1, 0)
        image_ch4 = np.maximum(image_ch4 - background_ch1, 0)
    #
    image_four_color = np.array([image_ch1, image_ch2, image_ch3, image_ch4]).transpose((2, 1, 0))
    image_processed = np.int64(np.dot(image_four_color, matrix))
    image_processed = np.maximum(image_processed, 0)
    image_processed = np.minimum(image_processed, 65535)
    #
    result_1.append(Im.fromarray(np.uint16(image_processed[:, :, 0])))
    result_2.append(Im.fromarray(np.uint16(image_processed[:, :, 1])))
    result_3.append(Im.fromarray(np.uint16(image_processed[:, :, 2])))
    result_4.append(Im.fromarray(np.uint16(image_processed[:, :, 3])))
# results of linear unmixing are generated as multi-tif for analysis using AQUACOSMOS or FV1000 Viewer
result_1[0].save(folder_name + '/result.tif', save_all=True, append_images=result_1[1:])
result_2[0].save(folder_name + '/result.tif', save_all=True, append_images=result_2[1:])
result_3[0].save(folder_name + '/result.tif', save_all=True, append_images=result_3[1:])
result_4[0].save(folder_name + '/result.tif', save_all=True, append_images=result_4[1:])
