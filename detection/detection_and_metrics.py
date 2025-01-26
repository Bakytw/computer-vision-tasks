import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================== 1 Classifier model ============================
def get_cls_model(input_shape=(40, 100, 1)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    n_rows, n_cols, n_channels = input_shape
    classification_model = nn.Sequential(
        nn.Conv2d(n_channels, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 3 * 10, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )
    return classification_model


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    model = get_cls_model((40, 100, 1))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    ds_train = TensorDataset(X, y)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)

    model.train()
    for _ in range(15):
        for images, labels in dl_train:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    # torch.save(model.state_dict(), "classifier_model.pt")
    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    detection_model = nn.Sequential(
        nn.Conv2d(1, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 256, (3, 10)),
        nn.ReLU(),
        nn.Conv2d(256, 2, 1),
    )
    detection_model.eval()
    with torch.no_grad():
        for i in range(12):
            detection_model[i] = cls_model[i]
        detection_model[12].weight.data = cls_model[13].weight.data.reshape((256, 64, 3, 10))    
        detection_model[14].weight.data = cls_model[15].weight.data.reshape((2, 256, 1, 1))
    
    return detection_model


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    detection_model.eval()
    detections = {}
    for filename, image in dictionary_of_images.items():
        img_detections = []
        shape = image.shape
        image = np.pad(image, ((0, 220 - shape[0]), (0, 372 - shape[1])))
        image = torch.from_numpy(image).reshape(1, 1, 220, 372)
        with torch.no_grad():
            output = detection_model(image)
        heatmap = output.cpu().detach().numpy()[0, 1, :, :]
        heatmap = heatmap[:shape[0] // 8, :shape[1] // 8]
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if heatmap[i, j] >= 0.2:
                    img_detections.append([i * 8, j * 8, 40, 100, heatmap[i, j].item()])
        detections[filename] = img_detections
    return detections


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    x1 = max(first_bbox[1], second_bbox[1])
    y1 = max(first_bbox[0], second_bbox[0])
    x2 = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
    y2 = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection_area = (x2 - x1) * (y2 - y1)
    iou = intersection_area / float(first_bbox[2] * first_bbox[3] + second_bbox[2] * second_bbox[3] - intersection_area)
    return iou


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    iou_thr = 0.5
    whole_tp = []
    all_detections = []
    all = 0
    for filename, detections in pred_bboxes.items():
        sorted_detections = sorted(detections, key=lambda key: key[4], reverse=True)
        gt_detections = gt_bboxes.get(filename, [])
        all += len(gt_detections)
        tp = []
        fp = []
        for detection in sorted_detections:
            max_iou = 0
            remove_box = None
            
            for gt_detection in gt_detections:
                cur_iou = calc_iou(detection[:4], gt_detection)
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    remove_box = gt_detection

            if max_iou >= iou_thr:
                gt_detections.remove(remove_box)
                tp.append(detection[4])
            else:
                fp.append(detection[4])
        whole_tp.extend(tp)
        all_detections.extend(tp + fp)

    tp = sorted(whole_tp, reverse=True)
    all_detections = sorted(all_detections, reverse=True)
    id = 0
    recalls = [0]
    precisions = [1]
    i = 0
    while i < len(all_detections) - 1:
        while i < len(all_detections) - 1 and all_detections[i] == all_detections[i + 1]:
            i += 1
        while id != len(tp) and tp[id] >= all_detections[i]:
            id += 1
        recalls.append(id / all)
        precisions.append(id / (i + 1))
        i += 1
    recalls.append(len(tp) / all)
    precisions.append(len(tp) / len(all_detections))
    auc = 0
    for i in range(1, len(precisions)):
        auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
    return auc

# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.1):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    suppressed = {}
    for filename, detections in detections_dictionary.items():
        suppressed[filename] = []
        sorted_detections = sorted(detections, key=lambda key: key[4], reverse=True)
        ious = np.array([[calc_iou(b1, b2) for b2 in sorted_detections] for b1 in sorted_detections])
        hold = np.full(len(sorted_detections), True)
        for i in range(len(sorted_detections)):
            if hold[i]:
                hold[i + 1:][ious[i, i + 1:] >= iou_thr] = False
        sorted_detections = np.array(sorted_detections)
        suppressed[filename] = sorted_detections[hold]
    return suppressed