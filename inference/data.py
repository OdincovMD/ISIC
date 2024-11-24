import pandas as pd
import os
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from resizeimage import resizeimage
import torch

def augmentations():
    """
    Создает список трансформаций для валидационного датасета, включая ресайз, нормализацию и добавление шума.
    
    Возвращает:
        list: Список трансформаций для применения к изображениям.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    resize = transforms.Lambda(
        lambda img: resizeimage.resize_cover(img, [288, 288], validate=False)
    )

    add_gaussian_noise = transforms.Lambda(
        lambda img: torch.randn_like(img) * 0.02 + img
    )

    to_tensor = transforms.ToTensor()

    val_transforms = [
        transforms.Compose([
            resize,
            to_tensor,
            add_gaussian_noise,
            normalize,
        ]),
    ]

    return val_transforms

class ImageFolderWithPaths(ImageFolder):
    """
    Расширение класса ImageFolder для возвращения путей к изображениям вместе с данными и метками.
    """
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]  # Получение пути к изображению
        return original_tuple + (path,)

def load_data(path):
    """
    Перемещает изображения в новые папки на основе классов, указанных в CSV-файле.

    Аргументы:
        path (str): Путь к корневой папке с файлами 'labels.csv' и 'images'.

    Возвращает:
        str: Путь к новой папке с отсортированными изображениями.
    """
    df = pd.read_csv(os.path.join(path, 'labels.csv'))
    class_names = sorted(df.benign_malignant.unique())
    data_root = 'data'

    os.makedirs(data_root, exist_ok=True)
    for class_name in class_names:
        path_class = os.path.join(data_root, class_name)
        os.makedirs(path_class, exist_ok=True)
        for image in df[df.benign_malignant == class_name].image_name:
            original_image_path = os.path.join(path, 'images', image)
            shutil.copy(original_image_path, path_class)
    return data_root

def main(path):
    """
    Функция для загрузки данных, применения аугментаций и создания DataLoader.

    Аргументы:
        path (str): Путь к корневой папке с файлами 'labels.csv' и 'images'.

    Возвращает:
        DataLoader: Загруженные данные с аугментациями.
    """
    data_root = load_data(path)
    dataset = ConcatDataset([
        ImageFolderWithPaths(data_root, val_transform)
        for val_transform in augmentations()
    ])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    return dataloader
