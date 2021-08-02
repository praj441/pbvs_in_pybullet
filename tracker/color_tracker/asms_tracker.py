import math
from . import histogram as hist
import copy
import numpy as np


class AsmsTracker:
    def __init__(self, _dimSize=4, _range=256):
        self.lastPosition = []
        self.im = None
        self.im_old = None

        self.q_hist = hist.Histogram()
        self.b_hist = hist.Histogram()
        self.q_orig_hist = None

        self.defaultWidth = 0.
        self.defaultHeight = 0.

        self.wAvgBg = 0.
        self.bound1 = 0.
        self.bound2 = 0.

        self.maxIter = 15

    def histMeanShiftIsotropicScale(self, x1, y1, x2, y2):
        w2 = (x2 - x1) / 2
        h2 = (y2 - y1) / 2

        borderX = 5.
        borderY = 5.

        cx = x1 + w2
        cy = y1 + h2

        y1hist = hist.Histogram()
        h0 = 1

        for ii in range(self.maxIter):
            wh = h0 * w2 + borderX
            hh = h0 * h2 + borderY
            rowMin = max(0, int(cy - hh))
            rowMax = min(self.im.shape[0] - 1, int(cy + hh))
            colMin = max(0, int(cx - wh))
            colMax = min(self.im.shape[1] - 1, int(cx + wh))

            self.extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist)

            batta_q = y1hist.computeSimilarity(self.q_orig_hist)
            batta_b = y1hist.computeSimilarity(self.b_hist)

            # mean shift vector
            m0 = 0.0
            m1x = 0.0
            m1y = 0.0

            wg_dist_sum = 0
            wk_sum = 0
            Sbg = 0
            Sfg = 0

            for i in range(rowMin, rowMax):
                tmp_y = pow((cy - i) / hh, 2)
                for j in range(colMin, colMax):
                    arg = pow((cx - j) / wh, 2) + tmp_y
                    if arg > 1:
                        continue

                    # likelihood weights
                    wqi = math.sqrt(self.q_orig_hist.getValue(self.im[i, j]) / y1hist.getValue(self.im[i, j]))
                    wbi = math.sqrt(self.b_hist.getValue(self.im[i, j]) / y1hist.getValue(self.im[i, j]))

                    ### weight used now
                    w = max(wqi / batta_q - wbi / batta_b, 0.0)
                    wg = w * (-1.0 * self.kernelProfile_EpanechnikovDeriv(arg))
                    dist = math.sqrt(((j - cx) / w2) ** 2 + ((i - cy) / h2) ** 2)

                    wg_dist_sum += wg * dist
                    wk_sum += w * (self.kernelProfile_Epanechnikov(arg))

                    ### weight used now
                    Sbg += y1hist.getValue(self.im[i, j]) if wqi < wbi else 0
                    Sfg += self.q_orig_hist.getValue(self.im[i, j])

                    m0 += wg
                    m1x += (j - cx) * wg
                    m1y += (i - cy) * wg

            xn_1 = m1x / m0 + cx
            yn_1 = m1y / m0 + cy

            # Rebularization
            reg1 = (self.wAvgBg - Sbg / Sfg)
            if math.fabs(reg1) > self.bound1:
                reg1 = self.bound1 if reg1 > 0 else -self.bound1

            reg2 = -(math.log(h0))
            if math.fabs(reg1) > self.bound1:
                reg2 = self.bound2 if reg2 > 0 else -self.bound2

            h_tmp = (1.0 - wk_sum / m0) * h0 + (1.0 / h0) * (wg_dist_sum / m0) + reg1 + reg2

            if ((xn_1 - cx) ** 2 + (yn_1 - cy) ** 2) < 0.1:
                break

            if not math.isinf(m0) and m0 > 0:
                cx = xn_1
                cy = yn_1
                h0 = 0.7 * h0 + 0.3 * h_tmp
                if borderX > 5:
                    borderX /= 3
                    borderY /= 3
            elif ii == 0:
                borderX *= 3
                borderY *= 3

        return cx, cy, h0, ii

    def init(self, img, x1, y1, x2, y2):
        # boundary checks
        y1 = max(0, y1)
        y2 = min(img.shape[0] - 1, y2)
        x1 = max(0, x1)
        x2 = min(img.shape[1] - 1, x2)

        self.preprocessImage(img)

        self.extractForegroundHistogram(x1, y1, x2, y2, self.q_hist)
        self.q_orig_hist = copy.deepcopy(self.q_hist)

        self.extractBackgroundHistogram(x1, y1, x2, y2, self.b_hist)
        b_weights = copy.deepcopy(self.b_hist)

        b_weights.transformToWeights()

        self.q_hist.multiplyByWeights(b_weights)

        self.lastPosition = [x1, y1, x2, y2]
        self.defaultWidth = x2 - x1
        self.defaultHeight = y2 - y1

        w2 = (x2 - x1) / 2.
        h2 = (y2 - y1) / 2.
        cx = x1 + w2
        cy = y1 + h2
        wh = w2 + 5.
        hh = h2 + 5.

        Sbg = 0.
        Sfg = 0.

        for i in range(y1, y2 + 1):
            tmp_y = ((cy - i) / hh) ** 2
            for j in range(x1, x2 + 1):
                arg = ((cx - j) / wh) ** 2 + tmp_y
                # likelihood weights
                wqi = 1.0
                # print(self.b_hist.getValue(self.im[i, j]), self.q_orig_hist.getValue(self.im[i, j]))
                wbi = math.sqrt(self.b_hist.getValue(self.im[i, j]) / self.q_orig_hist.getValue(self.im[i, j]))
                Sbg += self.q_orig_hist.getValue(self.im[i, j]) if wqi < wbi else 0.0
                Sfg += self.q_orig_hist.getValue(self.im[i, j])

        # self.wAvgBg = 0.5
        self.wAvgBg = max(0.1, min(Sbg / Sfg, 0.5))
        self.bound1 = 0.05
        self.bound2 = 0.1

    def track(self, img, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1

        self.im_old = copy.deepcopy(self.im)
        self.preprocessImage(img)

        # MS with scale estimation
        modeCenterX, modeCenterY, scale, _ = self.histMeanShiftIsotropicScale(x1, y1, x2, y2)
        width = 0.7 * width + 0.3 * width * scale
        height = 0.7 * height + 0.3 * height * scale

        # Forward-Backward validation
        if abs(math.log(scale)) > 0.05:
            tmp_im = self.im
            self.im = self.im_old
            _, _, scaleB, _ = self.histMeanShiftIsotropicScale(modeCenterX - width / 2, modeCenterY - height / 2,
                                                               modeCenterX + width / 2, modeCenterY + height / 2)
            self.im = tmp_im

            if abs(math.log(scale * scaleB)) > 0.1:
                alfa = 0.1 * (self.defaultWidth / float(x2 - x1))
                width = (0.9 - alfa) * (x2 - x1) + 0.1 * (x2 - x1) * scale + alfa * self.defaultWidth
                height = (0.9 - alfa) * (y2 - y1) + 0.1 * (y2 - y1) * scale + alfa * self.defaultHeight

        self.lastPosition = [modeCenterX - width / 2, modeCenterY - height / 2,
                             modeCenterX + width / 2, modeCenterY + height / 2]
        return modeCenterX - width / 2, modeCenterY - height / 2, modeCenterX + width / 2, modeCenterY + height / 2

    def update(self, img):
        return self.track(img, *self.lastPosition)

    def preprocessImage(self, img):
        self.im = copy.deepcopy(img.astype(np.uint8))

    def extractBackgroundHistogram(self, x1, y1, x2, y2, hist):
        offsetX = (x2 - x1) // 2
        offsetY = (y2 - y1) // 2

        # rows, cols, ch = image.shape
        rowMin = max(0, int(y1 - offsetY))
        rowMax = min(self.im.shape[0], int(y2 + offsetY + 1))
        colMin = max(0, int(x1 - offsetX))
        colMax = min(self.im.shape[1], int(x2 + offsetX + 1))

        numData = (rowMax - rowMin) * (colMax - colMin) - (y2 - y1) * (x2 - x1)

        if numData < 1:
            numData = (rowMax - rowMin) * (colMax - colMin) // 2 + 1

        d1 = []
        d2 = []
        d3 = []
        weights = []
        for y in range(rowMin, rowMax):
            for x in range(colMin, colMax):
                if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                    continue
                d1.append(self.im[y, x, 0])
                d2.append(self.im[y, x, 1])
                d3.append(self.im[y, x, 2])
        hist.clear()
        hist.insertValues(d1, d2, d3, weights)

    def extractForegroundHistogram(self, x1, y1, x2, y2, hist):
        hist.clear()
        d1 = []
        d2 = []
        d3 = []
        weights = []
        numData = int((y2 - y1) * (x2 - x1))
        if numData <= 0:
            return
        w2 = (x2 - x1) // 2
        h2 = (y2 - y1) // 2
        cx = x1 + w2
        cy = y1 + h2
        wh_i = 1.0 / (w2 * 1.4142 + 1)
        hh_i = 1.0 / (h2 * 1.4142 + 1)
        for y in range(y1, y2 + 1):
            tmp_y = pow((cy - y) * hh_i, 2)
            for x in range(x1, x2 + 1):
                d1.append(self.im[y, x, 0])
                d2.append(self.im[y, x, 1])
                d3.append(self.im[y, x, 2])
                weights.append(self.kernelProfile_Epanechnikov(pow((cx - x) * wh_i, 2) + tmp_y))

        hist.clear()
        hist.insertValues(d1, d2, d3, weights)

    def kernelProfile_Epanechnikov(self, x):
        return (2.0 / 3.14) * (1 - x) if x <= 1 else 0

    def kernelProfile_EpanechnikovDeriv(self, x):
        return -2.0 / 3.14 if x <= 1 else 0
