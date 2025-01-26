# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torch.nn.functional
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# !Этих импортов достаточно для решения данного задания
import cv2

CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NETWORK_SIZE=(64, 64)

class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = []
        for root_folder in root_folders:
            for folder in os.listdir(root_folder):
                path = os.path.join(root_folder, folder)
                for filename in os.listdir(path):
                    self.samples.append((os.path.join(path, filename), self.class_to_idx[folder]))
        self.classes_to_samples = {i: [] for i in range(len(self.classes))}
        for i, (img_path, class_idx) in enumerate(self.samples):
            self.classes_to_samples[class_idx].append(i)

        self.augmentations = A.Compose([
            A.Rotate(limit=30),
        ])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=NETWORK_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path, class_idx = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        aug_img_pts = self.augmentations(image=np.array(image, dtype=np.uint8))
        aug_image = aug_img_pts["image"]
        image = Image.fromarray(aug_image.astype(np.uint8))
        image = self.transform(image)
        return image, img_path, class_idx

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, 'r') as f:
            class_to_idx = {name: info["id"] for name, info in json.load(f).items()}
        return list(class_to_idx.keys()), class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)



class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.samples = []
        for image in os.listdir(root):
            self.samples.append(image)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=NETWORK_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.targets = None
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        if annotations_file is not None:
            self.targets = {}
            with open(annotations_file, 'r') as f:
                next(f)
                for line in f:
                    img_path, class_name = line.strip().split(',')
                    self.targets[img_path] = self.class_to_idx[class_name]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        img_path = self.samples[index]
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        image = self.transform(image)
        class_idx = self.targets.get(img_path, -1) if self.targets else -1
        return image, img_path, class_idx
    
    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, 'r') as f:
            class_to_idx = {name: info["id"] for name, info in json.load(f).items()}
        return list(class_to_idx.keys()), class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = 1024,
        transfer = False #ПОМЕНЯТЬ
    ):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=transfer)
        old_in_features = model.fc.in_features
        self.features_criterion = features_criterion
        if self.features_criterion:
            self.features_criterion = self.features_criterion(3)
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(old_in_features, internal_features)
            )
            for child in list(model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        else:
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(old_in_features, internal_features),
                torch.nn.ReLU(),
                torch.nn.Linear(internal_features, CLASSES_CNT)
            )
            for child in list(model.children())[:-3]:
                for param in child.parameters():
                    param.requires_grad = False
        self.model = model.to(DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        return self.forward(x).argmax(dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
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
        x, _, y = batch
        logits = self.forward(x)
        metrics = {}

        if self.features_criterion:
            loss = self.features_criterion(logits, y)
            metrics[f"train_loss"] = loss
            self.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                on_step="train" == "train",
                on_epoch=True,
            )
            return loss
        loss = self.loss_fn(logits, y)
        accs = (logits.argmax(dim=1) == y).sum() / y.shape[0]
        if loss is not None:
            metrics[f"train_loss"] = loss
        if accs is not None:
            metrics[f"train_accs"] = accs
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step="train" == "train",
            on_epoch=True,
        )
        return loss




def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train'],
        path_to_classes_json='./classes.json'
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
    )
    model = CustomNetwork(transfer=True)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "simple_model.pth")
    return model


def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ds_test = TestData(test_folder, path_to_classes_json)
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    model.eval()
    results = []
    for image, img_path, class_idx in dl_test:
        img_class = model.predict(image.to('cpu')).cpu().detach().numpy().ravel().item()
        results.append({'filename': img_path[0], 'class': ds_test.classes[img_class]})
    return results


def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    path_to_classes_json='./classes.json'
    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            reader = csv.DictReader(fhandle)
            for row in reader:
                res[row["filename"]] = row["class"]
        return res

    def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
        ok_cnt = 0
        all_cnt = 0
        for t, p in zip(y_true, y_pred):
            if cur_type == "all" or class_name_to_type[t] == cur_type:
                all_cnt += 1
                if t == p:
                    ok_cnt += 1
        return ok_cnt / max(1, all_cnt)

    results = apply_classifier(model, test_folder, path_to_classes_json)
    gt = read_csv(annotations_file)
    y_pred = [elem['class'] for elem in results]
    y_true = [gt[elem['filename']] for elem in results]
    
    with open(path_to_classes_json, 'r') as f:
        class_to_type = {name: info["type"] for name, info in json.load(f).items()}
    
    total_acc = calc_metric(y_true, y_pred, 'all', class_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_to_type)
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
        self.paths = []
        for image in os.listdir(background_path):
            self.paths.append(os.path.join(background_path, image))

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy
    
    @staticmethod
    def resize_icon(icon: np.ndarray) -> np.ndarray:
        size = np.random.randint(16, 128)
        return cv2.resize(icon, (size, size))

    @staticmethod
    def pad_icon(icon: np.ndarray) -> np.ndarray:
        h, w = icon.shape[:2]
        pad_percentage = random.randint(0, 15) / 100
        pad_w = int(w * pad_percentage)
        pad_h = int(h * pad_percentage)
        return np.pad(icon, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    @staticmethod
    def change_color_icon(icon: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(icon[:, :, :3], cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = np.random.randint(0, 256)
        icon[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return icon

    @staticmethod
    def rotate_icon(icon: np.ndarray) -> np.ndarray:
        angle = random.randint(-15, 15)
        r, c = icon.shape[:2]
        return cv2.warpAffine(icon, cv2.getRotationMatrix2D((c / 2, r / 2), angle, 1), (c, r))

    @staticmethod
    def blur_icon(icon: np.ndarray) -> np.ndarray:
        kernel = np.zeros((3, 3))
        kernel[1, 0] = 1
        angle = random.randint(-90, 90)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((3 / 2, 3 / 2), angle, 1), (3, 3))
        return cv2.filter2D(icon, -1, kernel)

    @staticmethod
    def gauss_icon(icon: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(icon, (3,3), 3)

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        icon = np.array(Image.open(icon).convert("RGBA"))
        icon = self.resize_icon(icon)
        icon = self.pad_icon(icon)
        icon = self.change_color_icon(icon)
        icon = self.rotate_icon(icon)
        icon = self.blur_icon(icon)
        icon = self.gauss_icon(icon)
        
        bg = cv2.imread(self.paths[random.randint(0, len(self.paths) - 1)])
        
        h, w = icon.shape[:2]
        x = random.randint(0, bg.shape[1] - w)
        y = random.randint(0, bg.shape[0] - h)
        bg = bg[y : y + h, x : x + w]
        mask = icon[:, :, 3]
        icon = icon[:, :, :3]
        bg[mask > 0] = icon[mask > 0]
        return bg


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    icon_path, out_dir, background_path, n, icon = args
    generator = SignGenerator(background_path)
    out_dir = os.path.join(out_dir, icon[:-4])
    for i in range(n):
        image = generator.get_sample(icon_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = os.path.join(out_dir, f'{i:06}.png')
        cv2.imwrite(os.path.join(out_dir, filename), image)


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    for image in os.listdir(icons_path):
        os.mkdir(os.path.join(output_folder, image[:-4]))
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
                icon_file
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train', './gen'],
        path_to_classes_json='./classes.json'
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
    )
    model = CustomNetwork(transfer=True)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "simple_model_with_synt.pth")
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """
        f1 = outputs[0]
        f2 = outputs[1]
        y1 = labels[0]
        y2 = labels[1]
        m = self.margin
        eps = self.eps
        dist_pos = (f2 - f1).square().sum(axis=-1)
        dist_neg = torch.nn.functional.relu(m - (dist_pos + eps).sqrt()).square()
        loss = 0.5 * torch.where(y1 == y2, dist_pos, dist_neg)
        return loss.mean() 

class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        while True:
            idxs = []
            for sign_class in np.random.choice(len(self.data_source.classes), self.classes_per_batch, replace=False):
                idx = np.random.choice(self.data_source.classes_to_samples[sign_class], self.elems_per_class)
                idxs += list(idx)
            yield idxs

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        return len(self.data_source) // (self.classes_per_batch * self.elems_per_class)


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    ds_train = DatasetRTSD(
        root_folders=['./cropped-train', './gen'],
        path_to_classes_json='./classes.json'
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        batch_sampler=CustomBatchSampler(ds_train, 4, 32)
    )
    model = CustomNetwork(features_criterion=FeaturesLoss, transfer=True)
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), "improved_features_model.pth")
    return model


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.eval()
        self.n_neighbors = n_neighbors

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        ### YOUR CODE HERE

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        ### YOUR CODE HERE

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        ### YOUR CODE HERE - предсказание нейросетевой модели
        features, model_pred = ...
        features = features / np.linalg.norm(features, axis=1)[:, None]
        ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = ...
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        self.data_source = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        idxs = []
        for sign_class in range(len(self.data_source.classes)):
            id = np.random.choice(self.data_source.classes_to_samples[sign_class], size=self.examples_per_class, replace=False)
            idxs.extend(id)
        return iter(idxs)

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        return self.class_count * self.examples_per_class


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE


if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.
    pass
