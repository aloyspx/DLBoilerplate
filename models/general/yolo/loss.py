import torch
from torch import nn

from models.general.yolo.utils import iou


# prediction (batch x S x S x 100)
# [
# 0->89: classes,
# 90->93: box1,
# 94: box1 confidence
# 95->98: box2,
# 99: box2 confidence
# ]

# label (batch x S x S x 95)
# [
# 0->89: classes,
# 90->93: box,
# 94: object probability in box = 1
# ]

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord, self.lambda_noobj = 5.0, 0.5

    def forward(self, output, label):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # iou for box proposition 1 of batches
        iou_b1 = iou(output[..., 90:94], label[..., 90:94])
        # iou for box proposition 2 of batches
        iou_b2 = iou(output[..., 95:99], label[..., 90:94])
        # concatenate box1 and box2 ious (batches, patches, patches, 2 ious)
        ious = torch.cat([iou_b1.unsqueeze(-1), iou_b2.unsqueeze(-1)], dim=-1)
        # best box indices
        best_box = torch.argmax(ious, dim=-1)
        # P(object), shape: [B, S, S, 1]. Is an object present in this patch?
        # Can be 0 if no object in box
        is_object = label[..., 94].unsqueeze(3)

        ## Localization Loss ##
        ## If there is an object in the patch,
        pred_loc_idx = torch.Tensor([[90, 91], [95, 96]])[best_box].type(torch.int64).to(device)
        local_loss = torch.nn.functional.mse_loss(
            (is_object * output.gather(-1, pred_loc_idx)).flatten(end_dim=-2),
            (is_object * label[..., 90:92]).flatten(end_dim=-2)
        )

        pred_size_idx = torch.Tensor([[92, 93], [97, 98]])[best_box].type(torch.int64).to(device)
        sizes = output.gather(-1, pred_size_idx)
        size_loss = torch.nn.functional.mse_loss(
            (is_object * torch.sign(sizes) * torch.sqrt(torch.abs(sizes))).flatten(end_dim=-2),
            (is_object * torch.sqrt(label[..., 92:94])).flatten(end_dim=-2)
        )

        ## Classification Loss ##
        # We take the mean squared error of every (batch x S x S) x classes but only if an object is present.
        # Remember that each class probability [1:90] is a conditional probability P(Class c_i | object)
        # The P(Object) is [90] in the label. So to filter out those with no objects, we remove them.
        class_loss = torch.nn.functional.mse_loss(
            (is_object * output[..., 0:90]).flatten(end_dim=-2),
            (is_object * label[..., 0:90]).flatten(end_dim=-2),
            reduction="sum")

        ## Confidence Loss ##
        # Very generally, P(object in the box | object)
        # Remember that this confidence is C = P(object) * IoU_{truth-pred}
        # To specialize each box predictor (2 of them), we only want one bounding box predictor to be responsible
        # for each object. So we only calculate for the one with highest IoU_{truth-pred}.
        pred_conf_idx = torch.Tensor([94, 99])[best_box].type(torch.int64).unsqueeze(-1).to(device)
        obj_conf_loss = torch.nn.functional.mse_loss(
            (is_object * output.gather(-1, pred_conf_idx)).flatten(end_dim=-2),
            (is_object * label[..., 94].unsqueeze(-1)).flatten(end_dim=-2),
            reduction="sum"
        )

        # Very generally, P(no object in the box | no object)
        # Here we use both boxes
        no_obj_conf_loss = torch.nn.functional.mse_loss(
            ((1 - is_object) * output[..., [94, 99]]).flatten(end_dim=-2),
            ((1 - is_object) * label[..., 94].unsqueeze(-1).repeat(1, 1, 1, 2)).flatten(end_dim=-2),
            reduction="sum"
        )

        del pred_loc_idx
        del pred_size_idx
        del pred_conf_idx

        return self.lambda_coord * local_loss + self.lambda_coord * size_loss + obj_conf_loss \
               + self.lambda_noobj * no_obj_conf_loss + class_loss
