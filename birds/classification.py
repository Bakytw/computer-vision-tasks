from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import torchvision.transforms as T
import torchvision
import PIL.Image
import random
from torch import nn
from tqdm.auto import tqdm
from numpy import array
import albumentations as A
import pytorch_lightning as L
import torchmetrics
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint


NETWORK_SIZE = (256, 256)
NUM_CLASSES = 50
BATCH_SIZE = 16
BASE_LR = 5e-4
MAX_EPOCHS = 30
BASE_WEIGHT_DECAY = 1e-6
NUM_WORKERS = os.cpu_count()
DEFAULT_TRANSFORM = T.Compose([
        T.Resize(size=NETWORK_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
AUG = A.Compose([
        A.Rotate(limit=30),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.OneOf([
            A.CLAHE(p=0.3),
            A.Blur()
        ])
])

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
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res


class BirdDataset(Dataset):
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
            for i, item in enumerate(train_gt.items()):
                img_jpg, label = item
                img_path = os.path.join(imgs_path, img_jpg)

                if mode == "train" and (i % 50) < int(train_fraction * 50):
                    data.append((img_path, label))
                elif mode == "valid" and (i % 50) >= int(train_fraction * 50):
                    data.append((img_path, label))

            rng = random.Random(split_seed)
            rng.shuffle(data)

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
            image = self._transform(image)
            return image, os.path.basename(img_path)
        else:
            img_path, label = self._data[index]
            image = PIL.Image.open(img_path).convert("RGB")
            if self._mode == "train":
                aug_img_pts = AUG(image=np.array(image, dtype=np.uint8))
                aug_image = aug_img_pts["image"]
                image = PIL.Image.fromarray(aug_image.astype(np.uint8))
            image = self._transform(image)
            assert image.shape[0] == 3
            return image, label


def get_my_model(num_classes=NUM_CLASSES, transfer=True):
    # model = torchvision.models.mobilenet_v2(pretrained=transfer)
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT if transfer else None
    model = torchvision.models.efficientnet_b3(weights=weights)
    old_p = model.classifier[0].p
    old_in_features = model.classifier[1].in_features
    old_out_features = model.classifier[1].out_features
    model.classifier = nn.Sequential(
        nn.Linear(old_in_features, old_out_features),
        nn.BatchNorm1d(old_out_features),
        nn.LeakyReLU(),
        nn.Dropout(old_p),
        nn.Linear(old_out_features, num_classes)
    )
    for child in list(model.children())[:-5]:
        for param in child.parameters():
            param.requires_grad = False

    return model

# За основу модуля берем модель с 6-го семинара
class BirdsClassifier(L.LightningModule):
    def __init__(self, *, num_classes = NUM_CLASSES, transfer=True, lr=BASE_LR, weight_decay=BASE_WEIGHT_DECAY, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.transfer = transfer
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.model = self.get_model()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
        )

    def get_model(self):
        return get_my_model(self.num_classes, self.transfer)

    def forward(self, x):
      return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")

    def _step(self, batch, kind):
        x, y = batch
        p = self.model(x)
        loss = self.loss_fn(p, y)
        accs = self.accuracy(p.argmax(axis=-1), y)
        return self._log_metrics(loss, accs, kind)

    def _log_metrics(self, loss, accs, kind):
        metrics = {}
        if loss is not None:
            metrics[f"{kind}_loss"] = loss
        if accs is not None:
            metrics[f"{kind}_accs"] = accs
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=kind == "train",
            on_epoch=True,
        )
        return loss

def train_model(
    model,
    dl_train,
    dl_valid,
    max_epochs=MAX_EPOCHS,
    fast_train=True
):
    callbacks = None
    logger = False
    enable_checkpointing = False
    if not fast_train:
        callbacks = [
            TQDMProgressBar(leave=True),
            LearningRateMonitor(),
            ModelCheckpoint(
                filename="{epoch}-{valid_accs:.3f}",
                monitor="valid_accs",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
        ]
        enable_checkpointing = None
        logger = None

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        max_epochs=max_epochs
    )
    trainer.fit(model, dl_train, dl_valid)
    return model

def train_classifier(train_gt, train_img_dir, fast_train=True):
    ds_train = BirdDataset(
        mode="train",
        train_gt=train_gt,
        imgs_path=train_img_dir
    )

    ds_valid = BirdDataset(
        mode="valid",
        train_gt=train_gt,
        imgs_path=train_img_dir
    )
    if fast_train:
        MAX_EPOCHS = 1
        NUM_WORKERS = 1
        BATCH_SIZE = 2
    else:
        MAX_EPOCHS = 30

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    dl_valid = DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    model = train_model(
        model=BirdsClassifier(transfer = not fast_train),
        dl_train=dl_train,
        dl_valid=dl_valid,
        max_epochs=MAX_EPOCHS,
        fast_train=fast_train
    )

    return model

def classify(model_filename, test_img_dir):
    ds_test = BirdDataset(
        mode="test",
        imgs_path=test_img_dir
    )

    dl_test = DataLoader(
        ds_test,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )
    model = BirdsClassifier(transfer=False).to(DEVICE)
    sd = torch.load(
        model_filename,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(sd)
    img_classes = {}
    model.eval()
    for image, img_path in dl_test:
        img_class = model(image.to(DEVICE)).cpu().detach().numpy()
        img_classes[img_path[0]] = np.argmax(img_class)
    return img_classes


# train_gt=read_csv(os.path.join("/content/drive/MyDrive/birds", 'gt.csv'))
# train_img_dir="/content/drive/MyDrive/birds/images"
# train_gt=read_csv(os.path.join("/home/bakyt/Prog/CV/birds/tests/00_test_img_input/train", 'gt.csv'))
# train_img_dir="/home/bakyt/Prog/CV/birds/tests/00_test_img_input/train/images"

# train_classifier(train_gt, train_img_dir, False)