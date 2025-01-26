from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import torchvision.transforms as T
import PIL.Image
import random
from torch import nn
from tqdm.auto import tqdm
from numpy import array
import albumentations as A

NETWORK_SIZE = (100, 100)
NUM_CLASSES = 28
BATCH_SIZE = 16
DEFAULT_TRANSFORM = T.Compose([
        T.Resize(size=NETWORK_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

AUG = A.Compose([
        A.Rotate(limit=15),
        A.Equalize(p=0.3),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using the GPU 😊")
else:
    DEVICE = torch.device("cpu")
    print("Using the CPU 😞") 

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

class FaceDataset(Dataset):
    def __init__(
            self,
            mode,
            train_gt = None,
            imgs_path = None,
            train_fraction=0.8,
            split_seed=42,
            transform = None
        ):

        data = []
        if mode == "test":
            for img_jpg in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, img_jpg)
                data.append(img_path)
        else:
            for item in train_gt.items():
                img_jpg, points = item
                img_path = os.path.join(imgs_path, img_jpg)
                points = np.array(points, dtype=np.float32)
                data.append((img_path, points))
            split = int(train_fraction * len(data))
            rng = random.Random(split_seed)
            rng.shuffle(data)
            if mode == "train":
                data = data[:split]
            elif mode == "valid":
                data = data[split:]

        print(len(data))
        self._data = data
        self._mode = mode
        if transform is None:
            transform = DEFAULT_TRANSFORM
        self._transform = transform


    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if self._mode == "test":
            img_path = self._data[index]
            image = PIL.Image.open(img_path).convert("RGB")
            sz = image.size
            image = self._transform(image)
            return os.path.basename(img_path), image, sz
        else:
            img_path, points = self._data[index]
            image = PIL.Image.open(img_path).convert("RGB")
            if self._mode == "train":
                aug_img_pts = AUG(image=np.array(image, dtype=np.uint8), keypoints=points.reshape(-1, 2))
                aug_image, aug_points = aug_img_pts["image"], aug_img_pts["keypoints"]
                image = PIL.Image.fromarray(aug_image.astype(np.uint8))
                points = np.array(aug_points, dtype=np.float32).flatten()
            points[::2] = (100 / image.size[0]) * points[::2]
            points[1::2] = (100 / image.size[1]) * points[1::2]
            image = self._transform(image)
            assert image.shape[0] == 3
            return image, points


class MyModel(nn.Sequential):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 10 * 10, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)


def train_detector(train_gt, train_img_dir, fast_train=True):
    ds_train = FaceDataset(
        mode="train",
        train_gt=train_gt,
        imgs_path=train_img_dir
    )
    ds_valid = FaceDataset(
        mode="valid",
        train_gt=train_gt,
        imgs_path=train_img_dir
    )


    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count(),
    )

    model = MyModel(NUM_CLASSES).to(DEVICE)
    loss_fn = nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-6)
    min_val_loss = 20
    if fast_train:
        EPOCHES = 1
    else:
        EPOCHES = 70

    for epoch in range(EPOCHES):
        model.train()
        train_loss = []
        progress_train = tqdm(
            total=len(dl_train),
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for x_batch, y_batch in dl_train:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            p_batch = model(x_batch)
            loss = loss_fn(p_batch, y_batch)
            train_loss.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_train.update()
        progress_train.close()

        train_loss = torch.stack(train_loss).mean()
        print(
            f"Epoch {epoch},",
            f"train_loss: {train_loss.item():.8f}",
        )

        model = model.eval()
        valid_loss = []
        progress_valid = tqdm(
            total=len(dl_valid),
            desc=f"Epoch {epoch}",
            leave=False,
        )
        for x_batch, y_batch in dl_valid:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            with torch.no_grad():
                p_batch = model(x_batch)
            loss = loss_fn(p_batch, y_batch)
            valid_loss.append(loss.detach())
            progress_valid.update()
        progress_valid.close()

        mean = torch.mean(torch.tensor(valid_loss)).cpu()
        if mean < min_val_loss:
            min_val_loss = mean
            sd = model.state_dict()
            torch.save(sd, f"best_model_{epoch}.pt") 

        print(
            f"Epoch {epoch},",
            f"valid_loss: {mean:.8f}",
        )

    return model

def detect(model_filename, test_img_dir):
    ds_test = FaceDataset(
        mode="test",
        imgs_path=test_img_dir
    )

    dl_test = DataLoader(
        ds_test,
        shuffle=False,
        batch_size=1,
        num_workers=os.cpu_count(),
    )

    model = MyModel().to(DEVICE)
    sd = torch.load(
        model_filename,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(sd)
    detected_points = {}
    model.eval()

    for img_path, image, sz in dl_test:
        points = model(image.to(DEVICE)).cpu().detach().numpy().flatten()
        points[::2] = (sz[0] / 100) * points[::2]
        points[1::2] = (sz[1] / 100) * points[1::2]
        detected_points[img_path[0]] = points

    return detected_points

# train_detector(
#     train_gt=read_csv(os.path.join(Path().absolute() / "tests/00_test_img_input/train/", 'gt.csv')),
#     train_img_dir=Path().absolute() / "tests/00_test_img_input/train/images",
#     fast_train=False
# )
