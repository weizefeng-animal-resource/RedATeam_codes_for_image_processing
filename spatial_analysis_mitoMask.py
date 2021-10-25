import cv2
import numpy as np
from PIL import Image as Im
from Image_processing import tool


address_read = ''
address_save = address_read + '/mitoMask_del.tif'
address_save_cell_roi = address_read + '/roi_cell{0}_del.tif'
#
video_c = Im.open(address_read + '/result_CFP.tif')
video_y = Im.open(address_read + '/result_YFP.tif')
video_o = Im.open(address_read + '/result_OFP.tif')
video_r = Im.open(address_read + '/result_RFP.tif')
#
# generate mask images 
mask_rois = []
for index in range(video_o.n_frames):
    print(index)
    video_c.seek(index)
    video_y.seek(index)
    video_o.seek(index)
    video_r.seek(index)
    frame_c = np.float32(np.array(video_c))
    frame_y = np.float32(np.array(video_y))
    frame_o = np.float32(np.array(video_o))
    frame_r = np.float32(np.array(video_r))
    if index == 0:
        frame_shape = frame_o.shape
        mask_or = cv2.medianBlur(np.uint8(np.minimum((frame_o + frame_r) / 32.0, 255)), 5)
        for _ in range(3):
            mask_or = tool.erosion(mask_or)
        _, mask_or = cv2.threshold(mask_or, 0, 1, cv2.THRESH_OTSU)
        nRois, roiImage, _, _ = cv2.connectedComponentsWithStats(mask_or)
        print(nRois)
        for index_roi in range(1, nRois):
            mask_roi = (roiImage == index_roi)
            if np.sum(mask_roi) > 1000:
                mask_rois.append(mask_roi)
        mask_rois = np.uint8(mask_rois)
        for _ in range(8):
            for index_roi in range(len(mask_rois)):
                mask_rois[index_roi] = tool.dilation(mask_rois[index_roi])
        mitoMask = np.zeros((len(mask_rois), frame_shape[0], frame_shape[1]))
    #
    _, mask_cy = cv2.threshold(np.uint8(np.minimum((frame_y + frame_c) / 32.0, 255)), 0, 1, cv2.THRESH_OTSU)
    for index_roi in range(len(mitoMask)):
        mitoMask[index_roi] += np.float32(cv2.distanceTransform(1 - mask_cy * mask_rois[index_roi], cv2.DIST_L2, 5))
mitoMask /= video_o.n_frames
mitoMask = np.uint8(mitoMask)
mitoMask[mask_rois == 0] = 255
#
image_save = [Im.fromarray(np.uint8(mitoMask[index_roi])) for index_roi in range(len(mitoMask))]
image_save[0].save(address_save, save_all=True, append_images=image_save[1:])
#
# generate RGB images indicating mask images
for index_roi in range(len(mitoMask)):
    image_show = np.zeros((frame_shape[0], frame_shape[1], 3))
    image_show[(mitoMask[index_roi] == 0)] = np.array([255, 255, 255])
    image_show[(mitoMask[index_roi] <= 1) * (mitoMask[index_roi] > 0)] = np.array([0, 0, 255])
    image_show[(mitoMask[index_roi] <= 2) * (mitoMask[index_roi] > 1)] = np.array([0, 255, 255])
    image_show[(mitoMask[index_roi] <= 3) * (mitoMask[index_roi] > 2)] = np.array([0, 255, 0])
    image_show[(mitoMask[index_roi] <= 4) * (mitoMask[index_roi] > 3)] = np.array([255, 255, 0])
    image_show[(mitoMask[index_roi] <= 5) * (mitoMask[index_roi] > 4)] = np.array([255, 127, 0])
    image_show[(mitoMask[index_roi] <= 6) * (mitoMask[index_roi] > 5)] = np.array([255, 0, 0])
    image_show[(mitoMask[index_roi] <= 7) * (mitoMask[index_roi] > 6)] = np.array([255, 0, 255])
    image_show[(mitoMask[index_roi] > 7)] = np.array([63, 63, 63])
    image_show[mitoMask[index_roi] == 255] = np.array([0, 0, 0])
    cv2.imwrite(address_save_cell_roi.format(index_roi), np.uint8(image_show))
