import cv2
import numpy as np
import random
import time


def findMergeLine(_lines, _p0, _p1, _vertical):
    indz = 0
    for k in _p1:
        mark, mindis = minDistance(_lines[_p0], _lines[k], _vertical)
        if mark or mindis <= 10:
            return True, indz, k
        indz = indz + 1
    return False, 0, 0


def minDistance(_line_0, _line_1, _vertical):
    _x0_0, _y0_0, _x1_0, _y1_0, _, _ = _line_0
    _x0_1, _y0_1, _x1_1, _y1_1, _, _ = _line_1
    if _vertical:
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


def merge(_lines, _vertical):
    for i in range(0, len(_lines)):
        temp = []
        for j in range(i + 1, len(_lines)):
            line0 = _lines[i]
            line1 = _lines[j]
            if np.abs(line0[4] - line1[4]) <= 2 and np.abs(line0[5] - line1[5]) <= 10:
                temp.append(j)
        inds = []
        while temp:
            flage, index0, index1 = findMergeLine(_lines, i, temp, _vertical)
            if not flage:
                break
            line = np.array([_lines[i], _lines[index1]])
            if _vertical:
                y_arr = np.array([line[0, 1], line[0, 3], line[1, 1], line[1, 3]])
                start, end = np.argmin(y_arr), np.argmax(y_arr)
            else:
                x_arr = np.array([line[0, 0], line[0, 2], line[1, 0], line[1, 2]])
                start, end = np.argmin(x_arr), np.argmax(x_arr)
            _lines[i] = [line[start // 2, start % 2 * 2], line[start // 2, start % 2 * 2 + 1],
                         line[end // 2, end % 2 * 2], line[end // 2, end % 2 * 2 + 1], _lines[i][4], _lines[i][5]]
            inds.append(index1)
            temp.pop(index0)
        inds.sort(reverse=True)
        for index in inds:
            _lines.pop(index)
    return _lines


if __name__ == '__main__':
    start_time = time.time()
    gray = cv2.imread('01.png', 0)
    img = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(scale=1)
    # 执行检测结果
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
    outlines.extend(merge(horizontal, False))
    outlines.extend(merge(vertical, True))
    out = np.zeros_like(img, np.uint8)
    for a in outlines:
        x0, y0, x1, y1, _, _ = [int(val) for val in a]
        cv2.line(out, (x0, y0), (x1, y1),
                 (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1, cv2.LINE_AA)
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.imshow('out', out)
    end_time = time.time()
    print(end_time - start_time)
    cv2.waitKey()
    cv2.destroyAllWindows()
