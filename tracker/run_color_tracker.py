import cv2
import os
import fire
import glob
from color_tracker import asms_tracker as tracker

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


if __name__ == '__main__':
    fire.Fire(run)
