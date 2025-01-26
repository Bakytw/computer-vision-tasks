import numpy as np

def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    intersection_area = max(0, min(xmax1, xmax2) - max(xmin1, xmin2)) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    return intersection_area / ((xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection_area)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {det[0]: det[1:] for det in frame_obj}
        hyp_dict = {det[0]: det[1:] for det in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if obj_id in obj_dict and hyp_id in hyp_dict:
                iou = iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del obj_dict[obj_id]
                    del hyp_dict[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        iou_pairwise = []
        for obj_id, obj_bbox in obj_dict.items():
            for hyp_id, hyp_bbox in hyp_dict.items():
                iou_pairwise.append((obj_id, hyp_id, iou_score(obj_bbox, hyp_bbox)))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: Update matches with current matched IDs
        iou_pairwise = sorted(iou_pairwise, reverse=True)
        for obj_id, hyp_id, iou in iou_pairwise:
            if iou > threshold and obj_id in obj_dict and hyp_id in hyp_dict:
                dist_sum += iou
                match_count += 1
                del obj_dict[obj_id]
                del hyp_dict[hyp_id]
                matches[obj_id] = hyp_id


    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    n = 0
    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {det[0]: det[1:] for det in frame_obj}
        n += len(obj_dict)
        hyp_dict = {det[0]: det[1:] for det in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if obj_id in obj_dict and hyp_id in hyp_dict:
                iou = iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del obj_dict[obj_id]
                    del hyp_dict[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        iou_pairwise = []
        for obj_id, obj_bbox in obj_dict.items():
            for hyp_id, hyp_bbox in hyp_dict.items():
                iou_pairwise.append((obj_id, hyp_id, iou_score(obj_bbox, hyp_bbox)))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        # Step 6: Update matches with current matched IDs
        iou_pairwise = sorted(iou_pairwise, reverse=True)
        for obj_id, hyp_id, iou in iou_pairwise:
            if iou > threshold and obj_id in obj_dict and hyp_id in hyp_dict:
                dist_sum += iou
                match_count += 1
                del obj_dict[obj_id]
                del hyp_dict[hyp_id]
                if obj_id in matches and matches[obj_id] != hyp_id:
                    mismatch_error += 1
                matches[obj_id] = hyp_id

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        missed_count += len(obj_dict)
        false_positive += len(hyp_dict)


    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / n

    return MOTP, MOTA
