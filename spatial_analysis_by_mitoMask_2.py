import cv2
import numpy as np
import pandas as pd
from PIL import Image as Im


address_read = ''  # should be the same address inputed in spatial_analysis_mitoMask.py
address_save = 'result.csv'
#
count = 0
mask_tif = Im.open(address_read + '/mitoMask.tif')  # mitoMask should be generated using spatial_analysis_mitoMask.py
mask_rois = []
while True:
    try:
        mask_tif.seek(count)
        mask_rois.append(np.array(mask_tif))
        count += 1
    except EOFError:
        break
mask_rois = np.int64(np.array(mask_rois))
#
video_c = Im.open(address_read + '/result_CFP.tif')
video_y = Im.open(address_read + '/result_YFP.tif')
video_o = Im.open(address_read + '/result_OFP.tif')
video_r = Im.open(address_read + '/result_RFP.tif')
result = []
for index in range(video_o.n_frames):
    print(index)
    video_c.seek(index)
    video_y.seek(index)
    video_o.seek(index)
    video_r.seek(index)
    frame_c = np.uint16(np.array(video_c))
    frame_y = np.uint16(np.array(video_y))
    frame_o = np.uint16(np.array(video_o))
    frame_r = np.uint16(np.array(video_r))
    frame_c = np.float32(cv2.medianBlur(frame_c, 5))
    frame_y = np.float32(cv2.medianBlur(frame_y, 5))
    frame_o = np.float32(cv2.medianBlur(frame_o, 5))
    frame_r = np.float32(cv2.medianBlur(frame_r, 5))
    #
    _, mask_or = cv2.threshold(np.uint8(np.minimum((frame_o + frame_r) / 32.0, 255)), 0, 1, cv2.THRESH_OTSU)
    fret_or = frame_o / (frame_r + 0.01) * mask_or
    _, mask_cy = cv2.threshold(np.uint8(np.minimum((frame_c + frame_y) / 32.0, 255)), 0, 1, cv2.THRESH_OTSU)
    fret_cy = frame_y / (frame_c + 0.01) * mask_cy
    #
    result_frame = []
    #
    mask_temp = (mask_rois == 0) * mask_cy
    result_frame.append(np.sum(mask_temp * fret_cy, axis=(1, 2)) / (np.sum(mask_temp, axis=(1, 2)) + 0.01))
    #
    for dist in range(1, 11):
        mask_temp = (mask_rois > (dist - 1)) * (mask_rois <= dist) * mask_or
        result_frame.append(np.sum(mask_temp * fret_or, axis=(1, 2)) / (np.sum(mask_temp, axis=(1, 2)) + 0.01))
    result.append(result_frame)
result = np.array(result)
if len(result.shape) == 2:
    print('single cell')
    result_row = np.mean(result[24:84, :], axis=0) / np.mean(result[:24, :], axis=0)
elif len(result.shape) == 3:
    print('multi cells')
    result_row = np.mean(result[24:84, :, :], axis=0) / np.mean(result[:24, :, :], axis=0)
    result_row = result_row.T
df = pd.DataFrame(result_row)
df.to_csv(address_save, mode='a', header=False)
