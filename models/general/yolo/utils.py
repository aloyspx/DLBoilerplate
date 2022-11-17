import torch


def unpack(box):
    return box[:, :, :, 0] - box[:, :, :, 2] / 2, \
           box[:, :, :, 1] - box[:, :, :, 3] / 2, \
           box[:, :, :, 0] + box[:, :, :, 2] / 2, \
           box[:, :, :, 1] + box[:, :, :, 3] / 2


def iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = unpack(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = unpack(box2)

    right = torch.max(b1_x1, b2_x1)
    left = torch.min(b1_x2, b2_x2)
    top = torch.max(b1_y1, b2_y1)
    bot = torch.min(b1_y2, b2_y2)

    a = (left - right).clamp(0) * (bot - top).clamp(0)
    a1 = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    a2 = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

    return a / (a1 + a2 - a + 1e-6)
