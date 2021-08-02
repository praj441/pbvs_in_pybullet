import cv2
import os
# import fire
import glob
from color_tracker import asms_tracker as tracker
import numpy as np
from tqdm import tqdm
import time
asms = tracker.AsmsTracker()


def get_init_rect(gt_path):
    with open(gt_path, 'r') as f:
        line = f.readline()
    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.strip().split(','))
    x1 = min(x1, min(x2, min(x3, x4)))
    x2 = max(x1, max(x2, max(x3, x4)))
    y1 = min(y1, min(y2, min(y3, y4)))
    y2 = max(y1, max(y2, max(y3, y4)))
    return int(x1), int(y1), int(x2), int(y2)


def run(chosed_dataset):
    TEST_DATASETS = ['girl', 'fish1']

    if chosed_dataset not in TEST_DATASETS:
        raise NotImplementedError('%s is not add to the code!' % chosed_dataset)

    jpgs_path = glob.glob(os.path.join(chosed_dataset, "*.jpg"))
    gt_path = os.path.join(chosed_dataset, "groundtruth.txt")

    for i, jpg in enumerate(jpgs_path):
        raw_img = cv2.imread(jpg)
        if 0 == i:
            asms.init(raw_img, *get_init_rect(gt_path))
        else:
            (x1, y1, x2, y2) = asms.update(raw_img)
            cv2.rectangle(raw_img, (int(x1), int(y1)), (int(x2), int(y2)), [255, 0, 0], 4)
            text1 = "frame {}:  ".format(i)
            text2 = "X:{}, Y:{}, W:{}, H:{}".format(int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1))
            cv2.putText(raw_img, text1 + text2, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 0, 0], 2)
            cv2.imshow("demo", raw_img)
            if 27 == cv2.waitKey(1):
                break

class Tracker(object):
    def __init__(self,init_frame_path,gt_box):
        super(Tracker, self).__init__()
        initial_frame = cv2.imread(init_frame_path)
        # print(jpg_files)
        asms.init(initial_frame,int(gt_box[0]),int(gt_box[1]),int(gt_box[2]),int(gt_box[3]))
    def track_next(self,frame_path):
        frame = cv2.imread(frame_path)
        (x1, y1, x2, y2) = asms.update(frame)
        return (x1, y1, x2, y2)



if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../output/tracked.avi',fourcc, 20.0, (640,352))


    gt_box = np.loadtxt('../data/gt_bbox.txt')
    jpg_files = [n for n in os.listdir('../data') if
                       n.find('jpg') >= 0]

    initial_frame = cv2.imread('../data/img0.jpg')
    # print(jpg_files)
    asms.init(initial_frame,int(gt_box[0]),int(gt_box[1]),int(gt_box[2]),int(gt_box[3]))
    x1 = gt_box[0]
    y1 = gt_box[1]
    x2 = gt_box[2]
    y2 = gt_box[3]
    cv2.rectangle(initial_frame, (int(x1), int(y1)), (int(x2), int(y2)), [255, 0, 0], 4)
    text1 = "frame {}:  ".format(0)
    text2 = "X:{}, Y:{}, W:{}, H:{}".format(int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1))
    cv2.putText(initial_frame, text1 + text2, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 0, 0], 2)
    out.write(initial_frame)
    cv2.imwrite('../output/frame_{0}.jpg'.format(0),initial_frame)

    for i in tqdm(range(1,len(jpg_files))):
        if i > 180:
            break
        frame = cv2.imread('../data/img{0}.jpg'.format(i))
        # st = time.time()
        (x1, y1, x2, y2) = asms.update(frame)
        # print(i,time.time()-st)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), [255, 0, 0], 4)
        text1 = "frame {}:  ".format(i)
        text2 = "X:{}, Y:{}, W:{}, H:{}".format(int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1))
        cv2.putText(frame, text1 + text2, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, [255, 0, 0], 2)
        out.write(frame)
        cv2.imwrite('../output/frame_{0}.jpg'.format(i),frame)

    out.release()

