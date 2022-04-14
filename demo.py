import cv2
import numpy as np
import random
import time


class Merge:
    def __init__(self, _lines, _vertical):
        self.vertical = _vertical
        self.lines = _lines

    def findMergeLine(self, i, ks):
        indz = 0
        for k in ks:
            mark, mindis = self.minDistance(i, k)
            if mark or mindis <= 15:
                return True, indz, k
            indz = indz + 1
        return False, 0, 0

    def minDistance(self, i, j):
        _x0_0, _y0_0, _x1_0, _y1_0, _, _ = self.lines[i]
        _x0_1, _y0_1, _x1_1, _y1_1, _, _ = self.lines[j]
        if self.vertical:
            min_0, max_0 = min(_y0_0, _y1_0), max(_y0_0, _y1_0)
            min_1, max_1 = min(_y0_1, _y1_1), max(_y0_1, _y1_1)
            if min_1 <= _y0_0 <= max_1 or min_1 <= _y1_0 <= max_1 or min_0 <= _y0_1 <= max_0 or min_0 <= _y1_1 <= max_0:
                return True, 0
        else:
            min_0, max_0 = min(_x0_0, _x1_0), max(_x0_0, _x1_0)
            min_1, max_1 = min(_x0_1, _x1_1), max(_x0_1, _x1_1)
            if min_1 <= _x0_0 <= max_1 or min_1 <= _x1_0 <= max_1 or min_0 <= _x0_1 <= max_0 or min_0 <= _x1_1 <= max_0:
                return True, 0
        distance = np.zeros(4)
        distance[0] = np.sqrt((_x0_0 - _x0_1) ** 2 + (_y0_0 - _y0_1) ** 2)
        distance[1] = np.sqrt((_x1_0 - _x0_1) ** 2 + (_y1_0 - _y0_1) ** 2)
        distance[2] = np.sqrt((_x0_0 - _x1_1) ** 2 + (_y0_0 - _y1_1) ** 2)
        distance[3] = np.sqrt((_x1_0 - _x1_1) ** 2 + (_y1_0 - _y1_1) ** 2)
        return False, np.min(distance)

    def mergeLines(self):
        for i in range(0, len(self.lines)):
            temp = []
            for j in range(i + 1, len(self.lines)):
                line0 = self.lines[i]
                line1 = self.lines[j]
                if np.abs(line0[4] - line1[4]) <= 2 and np.abs(line0[5] - line1[5]) <= 11:
                    temp.append(j)
            inds = []
            while temp:
                flage, index0, index1 = self.findMergeLine(i, temp)
                if not flage:
                    break
                line = np.array([self.lines[i], self.lines[index1]])
                if self.vertical:
                    y_arr = np.array([line[0, 1], line[0, 3], line[1, 1], line[1, 3]])
                    start, end = np.argmin(y_arr), np.argmax(y_arr)
                else:
                    x_arr = np.array([line[0, 0], line[0, 2], line[1, 0], line[1, 2]])
                    start, end = np.argmin(x_arr), np.argmax(x_arr)
                self.lines[i] = [line[start // 2, start % 2 * 2], line[start // 2, start % 2 * 2 + 1],
                                 line[end // 2, end % 2 * 2], line[end // 2, end % 2 * 2 + 1], self.lines[i][4],
                                 self.lines[i][5]]
                inds.append(index1)
                temp.pop(index0)
            inds.sort(reverse=True)
            for index in inds:
                self.lines.pop(index)
        return self.lines


if __name__ == '__main__':
    start_time = time.time()
    gray = cv2.imread('01.png', 0)
    img = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(scale=1)
    dlines = lsd.detect(gray)
    lines = dlines[0].reshape(-1, 4)
    vertical = []
    horizontal = []
    for a in lines:
        x0, y0, x1, y1 = a
        theta = np.arctan((y1 - y0) / (x1 - x0 + np.spacing(1))) * 180 / np.pi
        if np.abs(theta) <= 45:
            b = y0 - (y1 - y0) / (x1 - x0) * x0
            horizontal.append([x0, y0, x1, y1, theta, b])
        else:
            b = x0 - (x1 - x0) / (y1 - y0) * y0
            if theta < 0:
                theta += 180
            vertical.append([x0, y0, x1, y1, theta, b])

    outlines = []
    outlines.extend(Merge(vertical, True).mergeLines())
    outlines.extend(Merge(horizontal, False).mergeLines())
    end_time = time.time()
    print(end_time - start_time)
    out = np.zeros_like(img, np.uint8)
    for a in outlines:
        x0, y0, x1, y1, _, _ = [int(val) for val in a]
        cv2.line(out, (x0, y0), (x1, y1),
                 (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, cv2.LINE_AA)
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.imshow('out', out)
    cv2.waitKey()
    cv2.destroyAllWindows()
