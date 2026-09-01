import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.no_losses = 0

    def predict(self):
        self.no_losses += 1
        return self.bbox

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        tracked_objects = []

        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det[:4]))
        else:
            iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)

            for t, trk in enumerate(self.trackers):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = iou(trk.bbox, det[:4])

            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))

            unmatched_tracks = set(range(len(self.trackers)))
            unmatched_detections = set(range(len(detections)))

            for t, d in matched_indices:
                if iou_matrix[t, d] < self.iou_threshold:
                    continue
                self.trackers[t].update(detections[d][:4])
                unmatched_tracks.discard(t)
                unmatched_detections.discard(d)

            for d in unmatched_detections:
                self.trackers.append(KalmanBoxTracker(detections[d][:4]))

            self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        for trk in self.trackers:
            if trk.hits >= self.min_hits:
                tracked_objects.append([
                    *trk.bbox,
                    trk.id
                ])

        return np.array(tracked_objects)
