import cv2
import numpy as np


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

jointColors = (
        (0, 165, 255), (0, 0, 255), (0, 0, 255), (255, 255, 255),
        (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 0),
        (255, 0, 0), (0, 255, 85), (170, 255, 0), (47,255,173),
        (0, 255, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255),
        (0, 0, 255)) 

Skcolors = (
        (255, 0, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (0, 255, 255), (0, 255, 255), (0, 0, 255),
        (0, 0, 255), (0, 255, 85), (170, 255, 0), (47,255,173),
        (0, 255, 0), (0, 0, 0), (0, 0, 0), (0, 0, 255),
        (0, 0, 255))             
        
def show_poses(img, poses, scores, pose_score_threshold=0.8, point_score_threshold=0.8, skeleton=None, draw_ellipses=False):
    if poses.size == 0:
        return img
    if skeleton is None:
        skeleton = default_skeleton
    stick_width = 4
    img_limbs = np.copy(img)
    for pose, pose_score in zip(poses, scores):
        if pose_score <= pose_score_threshold:
            continue
        points = pose[:, :2].astype(int).tolist()
        points_scores = pose[:, 2]
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)), angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, (0,255,255))
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=Skcolors[i], thickness=stick_width)
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img_limbs, tuple(p), 5, jointColors[i], -1)
                
    
    cv2.addWeighted(img, 0.4, img_limbs, 0.9, 0, dst=img)
    return img
