import torch.cuda
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.coco2yolo import Coco
from loss import YOLOLoss
from model import YOLO

transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = Coco(
    root="data/coco/train2017",
    annFile="data/coco/annotations/instances_train2017.json",
    transform=transform,
)

epochs = 200
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

net = YOLO()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=5e-4)
criterion = YOLOLoss()

if torch.cuda.is_available():
    net.to("cuda")

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

for epoch in range(epochs):
    running_loss = 0.0
    for i, (image, box) in enumerate(train_dataloader):
        net.zero_grad()

        if torch.cuda.is_available():
            image, box = image.to("cuda"), box.to("cuda")

        output = net(image)

        loss = criterion(output, box)
        loss.backward()

        running_loss += loss.item() / image.shape[0]

        if i % 10 == 0:
            print(f"iteration {i} / {len(train_dataloader)} : {loss.item() / image.shape[0]}")

        optimizer.step()

    print(f"Epoch {epoch}/{epochs}: {running_loss / len(train_dataloader)}")

torch.save({
    "model": net.state_dict(),
    "optimizer": optimizer.state_dict()
}, "checkpoint.pth")
