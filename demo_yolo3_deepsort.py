import os
import cv2
import time
import argparse
import numpy as np
import sys

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes


#Detector Faces
class DataBatch:
    pass


def NMS(boxes, overlap_threshold):
    '''

    :param boxes: np nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != np.float32:
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = np.argsort(sc)  # 从小到大排序

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


class Predict(object):

    def __init__(self,
                 mxnet,
                 symbol_file_path,
                 model_file_path,
                 ctx,
                 receptive_field_list,
                 receptive_field_stride,
                 bbox_small_list,
                 bbox_large_list,
                 receptive_field_center_start,
                 num_output_scales
                 ):
        self.mxnet = mxnet
        self.symbol_file_path = symbol_file_path
        self.model_file_path = model_file_path
        self.ctx = ctx

        self.receptive_field_list = receptive_field_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.num_output_scales = num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]
        self.input_height = 480
        self.input_width = 640
        self.__load_model()

    def __load_model(self):
        # load symbol and parameters
        print('----> load symbol file: %s\n----> load model file: %s' % (self.symbol_file_path, self.model_file_path))
        if not os.path.exists(self.symbol_file_path):
            print('The symbol file does not exist!!!!')
            sys.exit(1)
        if not os.path.exists(self.model_file_path):
            print('The model file does not exist!!!!')
            sys.exit(1)
        self.symbol_net = self.mxnet.symbol.load(self.symbol_file_path)
        data_name = 'data'
        data_name_shape = (data_name, (1, 3, self.input_height, self.input_width))
        self.module = self.mxnet.module.Module(symbol=self.symbol_net,
                                               data_names=[data_name],
                                               label_names=None,
                                               context=self.ctx,
                                               work_load_list=None)
        self.module.bind(data_shapes=[data_name_shape],
                         for_training=False)

        save_dict = self.mxnet.nd.load(self.model_file_path)
        self.arg_name_arrays = dict()
        self.arg_name_arrays['data'] = self.mxnet.nd.zeros((1, 3, self.input_height, self.input_width), self.ctx)
        self.aux_name_arrays = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})
        self.module.init_params(arg_params=self.arg_name_arrays,
                                aux_params=self.aux_name_arrays,
                                allow_missing=True)
        print('----> Model is loaded successfully.')

    def predict(self, image, resize_scale=1, score_threshold=0.8, top_k=100, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None

        bbox_collection = []

        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side

        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

        input_image = input_image.astype(dtype=np.float32)
        input_image = input_image[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])

        data_batch = DataBatch()
        data_batch.data = [self.mxnet.ndarray.array(input_image, self.ctx)]
        
        tic = time.time()
        self.module.forward(data_batch=data_batch, is_train=False)
        results = self.module.get_outputs()
        outputs = []
        for output in results:
            outputs.append(output.asnumpy()) 
        toc = time.time()
        infer_time = (toc - tic) * 1000

        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = np.squeeze(outputs[i * 2], (0, 1))

            # score_map_show = score_map * 255
            # score_map_show[score_map_show < 0] = 0
            # score_map_show[score_map_show > 255] = 255
            # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=np.uint8), (0, 0), fx=2, fy=2))
            # cv2.waitKey()

            bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = np.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_np = np.array(bbox_collection, dtype=np.float32)

        if NMS_flag:
            final_bboxes = NMS(bbox_collection_np, NMS_threshold)
            final_bboxes_ = []
            for i in range(final_bboxes.shape[0]):
                final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))

            # return final_bboxes_, infer_time
            return final_bboxes_
        else:
            # return bbox_collection_np, infer_time
            return bbox_collection_np


def bbox_to_xywh_cls_conf(bbox):
    # person_id = 1
    #confidence = 0.5
    # only person
    # bbox = bbox[person_id]

    if any(bbox[:, 4] > 0.5):

        bbox = bbox[bbox[:, 4] > 0.5, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

        return bbox[:, :4], bbox[:, 4]

    else:

        return None, None
        
class Detector(object):
    def __init__(self, args):
        self.args = args
        args.display = False
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # self.vdo = cv2.VideoCapture()
        # self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)
        # self.class_names = self.yolo3.class_names


    # def __enter__(self):
    #     assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
    #     self.vdo.open(self.args.VIDEO_PATH)
    #     self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     if self.args.save_path:
    #         fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    #         self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

    #     assert self.vdo.isOpened()
    #     return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def detect(self):
        sys.path.append('..')
        from face_detection.config_farm import configuration_10_560_25L_8scales_v1 as cfg
        import mxnet
        symbol_file_path = '/content/deep_sort_pytorch/face_detection/symbol_farm/symbol_10_560_25L_8scales_v1_deploy.json'
        model_file_path = '/content/deep_sort_pytorch/face_detection/saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params'
        my_predictor = Predict(mxnet=mxnet,
                           symbol_file_path=symbol_file_path,
                           model_file_path=model_file_path,
                           ctx=mxnet.gpu(0),
                           receptive_field_list=cfg.param_receptive_field_list,
                           receptive_field_stride=cfg.param_receptive_field_stride,
                           bbox_small_list=cfg.param_bbox_small_list,
                           bbox_large_list=cfg.param_bbox_large_list,
                           receptive_field_center_start=cfg.param_receptive_field_center_start,
                           num_output_scales=cfg.param_num_output_scales)

        cap = cv2.VideoCapture('/content/faces.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('/content/output_de.avi',fourcc, 20.0, (1600,900))
        while True: 
            start = time.time()
            _, ori_im = cap.read()
            # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            # bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
            bboxes = my_predictor.predict(im, resize_scale=1, score_threshold=0.3, 
                                            top_k=10000, NMS_threshold=0.3, 
                                            NMS_flag=True, skip_scale_branch_list=[])
            bboxes = np.array(bboxes)
            bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(bboxes)
            if bbox_xywh is not None:
                # select class person
                # mask = cls_ids==0
                # bbox_xcycwh = bbox_xcycwh[mask]
                # bbox_xcycwh[:,3:] *= 1.2

                # cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
            print(ori_im.shape)
            out.write(ori_im)
            # if self.args.display:
            #     cv2.imshow("test", ori_im)
            #     cv2.waitKey(1)

            # if self.args.save_path:
            #     self.output.write(ori_im)
            

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    # parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    # parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    # parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    # with Detector(args) as det:
    #     det.detect()
    det = Detector(args)
    det.detect()