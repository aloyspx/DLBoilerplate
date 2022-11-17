from typing import Optional, Callable

import torch
from torchvision.datasets.coco import CocoDetection


def xywh_to_cxcywh(bbox, r_w, r_h):
    # coco : [x_min, y_min, width, height]
    # yolo : [cx, cy, width, height]
    return [r_w * (bbox[0] + bbox[2] / 2), r_h * (bbox[1] + bbox[3] / 2), r_w * bbox[2], r_h * bbox[3]]


def which_patch(bbox, w, h):
    x, y = bbox[:2]
    a, b = int(7 * (x / w)), int(7 * (y / h))
    bbox = [x - a * (w / 7), y - b * (h / 7), *bbox[2:]]
    return bbox, a, b


class Coco(CocoDetection):

    def __init__(self, root: str, annFile: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, ):
        super(Coco, self).__init__(
            root,
            annFile,
            None,
            None,
            None)

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)

        org_w, org_h = image.size
        new_w, new_h = 448, 448
        r_w, r_h = new_w / org_w, new_h / org_h

        label = torch.zeros((7, 7, 100))
        bb1_filled = torch.zeros(7, 7)
        bb2_filled = torch.zeros(7, 7)

        for target in targets:
            cat = torch.nn.functional.one_hot(torch.Tensor([target["category_id"] - 1]).long(), num_classes=90) \
                .squeeze(0)
            bbox, a, b = which_patch(xywh_to_cxcywh(target["bbox"], r_w, r_h), new_w, new_h)

            if bb1_filled[a, b] == 0:
                label[a, b, :95] = torch.FloatTensor([*cat, *bbox, 1.0])
                bb1_filled[a, b] = 1
            elif bb2_filled[a, b] == 0:
                label[a, b, :90] = cat.float()
                label[a, b, 95:] = torch.FloatTensor([*bbox, 1.0])
            else:
                print("Error more than two boxes in the same patch.")
                raise RuntimeError

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
