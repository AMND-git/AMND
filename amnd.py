# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import math
import random
import itertools
import six.moves.cPickle as pickle
import chainer
import chainer.functions  as F


def competitive_learning(img, k):
    gamma = 0.015
    threshold = 100*math.sqrt(k)
    tmax = int((2*k-3) * threshold)
    img = img.reshape((-1, 1))
    clusters = [np.uint16(img.mean(axis = 0))]
    n = 1
    wc = [0] * k
    for i in range(tmax):
        x = img[random.randint(0, len(img)-1)]
        min_dist = np.linalg.norm(x-clusters[0])
        winner = 0
        for j in range(n):
            dist = np.linalg.norm(x-clusters[j])
            if dist < min_dist:
                min_dist = dist
                winner = j
        clusters[winner] = clusters[winner] + gamma*(x-clusters[winner]) 
        wc[winner] += 1
        if (wc[winner] >= threshold) and (n < k):
            n += 1
            clusters.append(clusters[winner])
            wc[n-1] = 0
            wc[winner] = 0
    clusters = sorted(clusters, key = lambda x: x[0])
    labels = [np.argmin(([np.linalg.norm(px-cluster) for cluster in clusters])) for px in img]
    return [clusters, labels]


def find_contours(img, count_holes=False):
    h, w = img.shape
    ret, thresh = cv2.threshold(img, 130, 240, cv2.THRESH_BINARY)
    img, all_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
    last = 0
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == -1:
            last = i
    contours = []
    hole_counts = []
    for i in range(len(hierarchy[0])):
        if all_contours[i].size <= 50:
            continue
        elif hierarchy[0][i][3] == last:
            x1, y1, w1, h1 = cv2.boundingRect(all_contours[i])
            if (x1 != 1) and (y1 != 1) and (x1+w1 != w-1) and (y1+h1 != h-1):
                contours.append(all_contours[i])
                if count_holes:
                    hole_counts.append(np.count_nonzero(hierarchy[0, :, 3] == i))
    if count_holes:
        return ([contours, hole_counts])
    return contours        


def clump_splitting(contour):
    approx = cv2.convexHull(contour, returnPoints = False)    
    defects = cv2.convexityDefects(contour, approx)
    if not defects.all():
        return False
    defects = list(defects)
    defects.sort(key=lambda i:i[0][3])
    defects.reverse()
    defects = defects[0:8]
    min_BN = 1
    for element in itertools.combinations(defects, 2):
        s1, e1, f1, d1 = element[0][0]
        s2, e2, f2, d2 = element[1][0]
        distance = np.linalg.norm(contour[f1][0]-contour[f2][0])
        length = min(cv2.arcLength(contour[min(f1, f2):max(f1, f2)+1], False),
                     cv2.arcLength(np.r_[contour[max(f1, f2):], contour[0:min(f1, f2)+1]], False))
        BN = distance / length
        CD1 = np.linalg.norm(contour[f1][0] - (contour[s1][0]+contour[e1][0])/2)
        CD2 = np.linalg.norm(contour[f2][0] - (contour[s2][0]+contour[e2][0])/2)
        SA = min(CD1, CD2) / (min(CD1, CD2)+distance)
        v1 = contour[f1][0] - (contour[s1][0]+contour[e1][0])/2
        v2 = contour[f2][0] - (contour[s2][0]+contour[e2][0])/2
        u12 = contour[f2][0]-contour[f1][0] 
        CC = math.pi - math.acos(round(np.dot(v1,v2) / 
                                (np.linalg.norm(contour[f1][0]-(contour[s1][0]+contour[e1][0])/2)
                                *np.linalg.norm(contour[f2][0]-(contour[s2][0]+contour[e2][0])/2)), 3))
        CL = max(math.acos(round(np.dot(v1, u12) /
                                (np.linalg.norm(contour[f1][0]-(contour[s1][0]+contour[e1][0])/2)
                                *np.linalg.norm(contour[f2][0]-contour[f1][0])), 3)),
                 math.acos(round(np.dot(v2,-u12) /
                                (np.linalg.norm(contour[f2][0]-(contour[s2][0]+contour[e2][0])/2)
                                *np.linalg.norm(contour[f2][0]-contour[f1][0])), 3)))
        fs1 = contour[s1][0]-contour[f1][0]
        fe1 = contour[e1][0]-contour[f1][0]
        fs2 = contour[s2][0]-contour[f2][0]
        fe2 = contour[e2][0]-contour[f2][0]
        CA1 = math.acos(round(np.dot(fs1, fe1)/(np.linalg.norm(fs1)*np.linalg.norm(fe1)), 3))
        CA2 = math.acos(round(np.dot(fs2, fe2)/(np.linalg.norm(fs2)*np.linalg.norm(fe2)), 3))
        if (CL < math.pi*7/18) and (CC < math.pi*7/12) and (SA > 0.12) and (max(CA1, CA2) < math.pi*1/2):
            if min_BN > BN:
                min_BN = BN
                ff1 = f1
                ff2 = f2
    if min_BN != 1:
        return [contour[ff1], contour[ff2]]
    else:
        return False


def classify(model, img):
    img = cv2.resize(img, (64, 64))
    img = np.array([img]).astype(np.float32).reshape((1, 1, 64, 64))/255
    x = chainer.Variable(img)
    h = F.max_pooling_2d(F.relu(model.bnorm1(model.conv1(x))), 2)
    h = F.max_pooling_2d(F.relu(model.bnorm2(model.conv2(h))), 2)
    h = F.max_pooling_2d(F.relu(model.bnorm3(model.conv2(h))), 2)
    h = F.dropout(F.relu(model.l3(h)))
    y = model.l4(h)
    if y.data[0][0] <= y.data[0][1]:
        return 1
    else:
        return 0
   
             
def identify(contour2, hole_count, x1, y1):
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    bounded2 = img[y1+y2-5:y1+y2+h2-5, x1+x2-5:x1+x2+w2-5]

    # identification with the trained CNN model
    if classify(model, bounded2):
        mask = np.full((h2, w2), 0, dtype = np.uint8)
        contour2 += np.array([-x2, -y2])
        cv2.drawContours(mask, [contour2], 0, 255, -1)
        extracted_bounded2 = cv2.bitwise_and(expanded_extracted_bounded[y2:y2+h2, x2:x2+w2],
                                             expanded_extracted_bounded[y2:y2+h2, x2:x2+w2], mask = mask)
        extracted_bounded2[extracted_bounded2 == 0] = 255
                          
        if (hole_count < 2) or \
           ((hole_count >= 2) and (not clump_splitting(contour2))):
            for i in range(y1+y2-3, y1+y2-3+h2):
                for j in range(x1+x2-3, x1+x2-3+w2):
                    if (result[i][j] == 255) and (extracted_bounded2[i-(y1+y2-3)][j-(x1+x2-3)] != 255):
                        result[i][j] = extracted_bounded2[i-(y1+y2-3)][j-(x1+x2-3)]

        # clump splitting
        else:            
            p1, p2 = clump_splitting(contour2)
            p1_index = np.where((contour2 == p1).all(axis = 2).flatten())[0]
            p2_index = np.where((contour2 == p2).all(axis = 2).flatten())[0]
            contours3 = [np.array(contour2[min(p1_index[0], p2_index[0]):max(p1_index[0], p2_index[0])+1]), 
                         np.array(list(contour2[0:(min(p1_index[0], p2_index[0])+1)]) + 
                                  list(contour2[max(p1_index[0], p2_index[0]):]))]
            for contour3 in contours3:
                mask = np.full((h2, w2), 0, dtype = np.uint8)
                cv2.drawContours(mask, [contour3], 0, 255, -1)
                extracted_bounded3 = cv2.bitwise_and(extracted_bounded2, extracted_bounded2, mask = mask)
                extracted_bounded3[extracted_bounded3 == 0] = 255                
                hole_count3 = find_contours(extracted_bounded3, True)[1]
                if len(hole_count3) == 0:
                    return
                contour3 += np.array([x2, y2])
                identify(contour3, hole_count3[0], x1, y1)
    else:
        return


# main
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open("model", "rb"))
samples = os.listdir("samples")
os.chdir("samples")
for sample in samples:
    path, ext = os.path.splitext(sample)
    if not ext in [".bmp", ".jpg", ".png", ".tiff"]:
        continue
    print ("Now processing: {}".format(sample))
    img = cv2.imread(sample, 0)
    h, w = img.shape
    
    # Normalization
    high = np.percentile(img, 99)
    low = np.percentile(img, 0.01)
    img[img >= high] = high
    img[img <= low] = low
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # Segmentation with competitive learning
    clusters, labels = competitive_learning(img, 3)
    extracted = np.array([255] * (h*w))
    extracted[np.array(labels) == 0] = 0
    extracted = extracted.reshape((h, w))
    h += 4
    w += 4
    expanded_img = np.full((h, w), 255, dtype = np.uint8)
    expanded_img[2:-2, 2:-2] = img
    expanded_extracted = np.full((h, w), 255, dtype = np.uint8)
    expanded_extracted[2:-2, 2:-2] = extracted

    contours = find_contours(expanded_extracted)
    result = np.full((h, w), 255, dtype = np.uint8)
    
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)

        # Segmentation with k-means algorithm
        bounded = expanded_img[y1:y1+h1, x1:x1+w1]
        bounded = np.float32(bounded.reshape((-1, 1)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        compactness, labels, centers = cv2.kmeans(bounded, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((h1, w1))
        extracted_bounded = np.full((h1, w1), 255, dtype = np.uint8)
        extracted_bounded[labels == np.argmin(centers)] = 1

        mask = np.full((h, w), 0, dtype = np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mask = mask[y1:y1+h1, x1:x1+w1]
        extracted_bounded = cv2.bitwise_and(extracted_bounded, extracted_bounded, mask = mask)
        extracted_bounded[extracted_bounded == 0] = 255
        expanded_extracted_bounded = np.full((h1+6, w1+6), 255, dtype = np.uint8)
        expanded_extracted_bounded[3:-3, 3:-3] = extracted_bounded
        contours2, hole_counts2 = find_contours(expanded_extracted_bounded, count_holes = True)

        # identification with the trained CNN model and clump splitting
        for i, contour2 in enumerate(contours2):
            identify(contour2, hole_counts2[i], x1, y1)

    cv2.imwrite("result_{}.png".format(path), result[3:-3, 3:-3])

print ("The process was done")
