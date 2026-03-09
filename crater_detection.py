from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import cv2
import numpy as np
import matplotlib.pyplot as plt

def crater_candidates(ml_img, ml_img_gray, blur_img, high):
    circles = cv2.HoughCircles(
        blur_img,
        cv2.HOUGH_GRADIENT,
        dp=1.3,         # 1.2       # 1.2
        minDist=18,     # 22        # 18
        param1=high,
        param2=10,      # 15        # 22
        minRadius=4,    # 8         # 4
        maxRadius=18,   # 22        # 18
    )

    candidates = []
    if circles is not None:
        candidates = np.round(circles[0]).astype(int).tolist()

    print(f"Circle candidates: {len(candidates)}")

    cand_img = cv2.cvtColor(ml_img_gray, cv2.COLOR_GRAY2RGB) # troubleshooting
    for (x, y, r) in candidates:
        cv2.circle(cand_img, (x, y), r, (255, 0, 0), 1)
    
    return candidates

def crater_features(img, edges, x, y, r):
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    inner = dist <= r * 0.6
    ring = (dist >= r * 0.8) & (dist <= r * 1.1)
    outer = (dist >= r * 1.2) & (dist <= r * 1.5)

    if ring.sum() < 10 or outer.sum() < 10:
        return None

    inner_mean = img[inner].mean()
    outer_mean = img[outer].mean()
    ring_edge = edges[ring].mean()
    ring_std = img[ring].std()
    contrast = outer_mean - inner_mean

    return [inner_mean, outer_mean, contrast, ring_edge, ring_std]

def extract_crater_features(ml_img, ml_edges, candidates):
    features = []
    valid_candidates = []
    for (x, y, r) in candidates:
        feats = crater_features(ml_img, ml_edges, x, y, r)
        if feats is not None:
            features.append(feats)
            valid_candidates.append((x, y, r))
    print("Valid candidates:", len(valid_candidates))
    
    features_arr = np.array(features)
    print(features_arr)
    return [features_arr, valid_candidates]

def train_detection(X, valid_candidates, img, path_for_new_img):
    X = np.array(X)
    crater_lst = []
    #print(X)
    if len(X) < 10:
        print("Not enough candidates to train. Try lowering Hough param2 or minRadius.")
        y = None
    else:
        edge_vals = X[:, 3]
        contrast_vals = X[:, 2]
        edge_thresh = np.percentile(edge_vals, 70)
        contrast_thresh = np.percentile(contrast_vals, 70)
        y = ((edge_vals > edge_thresh) & (contrast_vals > contrast_thresh)).astype(int)
        print("Crater labels (1s):", int(y.sum()), "of", len(y))

    if y is not None and len(np.unique(y)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)
        print(classification_report(y_test, pred, digits=3))

        scores = clf.predict_proba(scaler.transform(X))[:, 1]
        crater_img = img.copy()

        score_thresh = np.percentile(scores, 75)
        for (x, y0, r), score in zip(valid_candidates, scores):
            if score >= score_thresh:
                crater_lst.append([x, y0, r])
                cv2.circle(crater_img, (x, y0), r, (0, 255, 0), 2)

        cv2.imwrite(path_for_new_img, crater_img)
        return crater_lst
    else:
        print("Not enough labeled data to train. Try adjusting thresholds or Hough settings.")
        return []
        
def group_craters(crater_lst, ml_img, ml_img_ORIG, img_path): # group overlapping circles together
    OFFSET = 3
    combined_crater_lst = []
    row = 0
    for i in range(0, len(crater_lst)):
        if not crater_lst[i] in combined_crater_lst:
            combined_crater_lst.append([crater_lst[i]])
            crater_x = [crater_lst[i][0] - crater_lst[i][2] - OFFSET, crater_lst[i][0] + crater_lst[i][2] + OFFSET]
            crater_y = [crater_lst[i][1] - crater_lst[i][2] - OFFSET, crater_lst[i][1] + crater_lst[i][2] + OFFSET]
            for m in range(i+1, len(crater_lst)):
                new_crater_x = [crater_lst[m][0] - crater_lst[m][2] - OFFSET, crater_lst[m][0] + crater_lst[m][2] + OFFSET]
                new_crater_y = [crater_lst[m][1] - crater_lst[m][2] - OFFSET, crater_lst[m][1] + crater_lst[m][2] + OFFSET]
                # print(crater_x, crater_y)
                # print(new_crater_x, new_crater_y)
                # print(((crater_x[1] > new_crater_x[0] and crater_x[0] < new_crater_x[1]) or (crater_x[0] < new_crater_x[1] and crater_x[1] > new_crater_x[0])) and ((crater_y[1] > new_crater_y[0] and crater_y[0] < new_crater_y[1]) or (crater_y[0] < new_crater_y[1] and crater_y[1] > new_crater_y[0])))
                if ((crater_x[1] > new_crater_x[0] and crater_x[0] < new_crater_x[1]) or (crater_x[0] < new_crater_x[1] and crater_x[1] > new_crater_x[0])) and ((crater_y[1] > new_crater_y[0] and crater_y[0] < new_crater_y[1]) or (crater_y[0] < new_crater_y[1] and crater_y[1] > new_crater_y[0])):
                    combined_crater_lst[row].append(crater_lst[m])
        row += 1
    
    combined_crater_img = ml_img_ORIG.copy()
    mask = np.zeros_like(ml_img, dtype=np.uint8)
    for crater_group in combined_crater_lst:
        for x,y,r in crater_group:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)  # filled circle
        
    kernel = np.ones((7,7), np.uint8) # add a boundary
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(combined_crater_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(img_path, combined_crater_img)
    
    return contours

def crater_detection(img_path, ml_img_ORIG, ml_img):
    ml_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(ml_img)
    ml_blur = cv2.GaussianBlur(ml_clahe, (9, 9), 0)

    v = np.median(ml_blur)
    low = int(max(0, (1.0 - 0.33) * v))
    high = int(min(255, (1.0 + 0.33) * v))
    ml_edges = cv2.Canny(ml_blur, low, high)
    
    candidates = crater_candidates(ml_img_ORIG, ml_img, ml_blur, high)
    [features, valid_candidates] = extract_crater_features(ml_img, ml_edges, candidates)
    
    path_arr = img_path.split('/')
    img_name = path_arr[len(path_arr) - 1].split('.')[0] + '_craters.png'
    crater_path = path_arr[:len(img_path.split('/'))-1]
    crater_path = '/'.join(crater_path)
    crater_path += '/' + img_name
    crater_lst = train_detection(features, valid_candidates, ml_img_ORIG, crater_path)
    
    img_name_2 = path_arr[len(path_arr) - 1].split('.')[0] + '_combined_craters.png'
    combined_craters_path = path_arr[:len(img_path.split('/'))-1]
    combined_craters_path = '/'.join(combined_craters_path)
    combined_craters_path += '/' + img_name_2
    contours = group_craters(crater_lst, ml_img, ml_img_ORIG, combined_craters_path)
    
    return [combined_craters_path, contours]
    
