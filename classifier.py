import os
import torch
from torchvision import transforms
from PIL import Image
from resizeimage import resizeimage
import torch
import numpy as np
import models.WideResNet as WideResNet
import models.VotingClassifier as VotingClassifier
import models.YoloV8 as YoloV8


class Classifier:
    def __init__(self, device='cpu', voting='soft'):
        """
        Класс для предсказания метки.

        Аргументы:
            device (torch.device): Устройство для вычислений (CPU/GPU).
            voting (str): Метод голосования. Возможные значения:
                        - 'soft': Soft voting (усреднение вероятностей).
                        - 'majority': Majority voting (мода предсказаний).
        """
        
        self.models = WideResNet.main() \
            + VotingClassifier.main() \
            + YoloV8.main()
        self.device = device
        self.voting = voting

    def predict(self, image_path):
        """
        Обрабатывает одно изображение с использованием ансамбля моделей и возвращает предсказанный класс.

        Аргументы:
            image_path (str): Путь до изображения.
        Возвращает:
            str: Предсказанный класс.
        """
        image = self.preprocess(image_path)
        image = image.to(self.device)
        fold_preds = []
        fold_probs = []

        # Перевод всех моделей в режим оценки
        for model in self.models[:-1]:
            model.eval()

        with torch.no_grad():
            for i, model in enumerate(self.models[:-1]):
                if i != 4:  # Обработка специфики 4-й модели
                    preds = model(image.unsqueeze(0))  # Предсказания модели
                    probs = torch.softmax(preds, dim=1)  # Вероятности классов
                else:
                    probs = model.predict(image.unsqueeze(0))  # Вероятности напрямую

                fold_probs.append(probs.cpu().numpy())
                preds_class = probs.argmax(dim=1)
                fold_preds.append(preds_class.cpu().numpy())

            # Обработка последней модели
            result = self.models[-1](image_path, verbose=False)
            probs = result[0].probs.data.unsqueeze(0)
            fold_probs.append(probs.cpu().numpy())
            preds_class = probs.argmax(dim=1)
            fold_preds.append(preds_class.cpu().numpy())

            # Голосование
            if self.voting == 'majority':
                # Majority voting (мода по всем предсказаниям)
                preds_voted = np.stack(fold_preds, axis=1)
                final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds_voted)
            elif self.voting == 'soft':
                # Soft voting (усреднение вероятностей)
                avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
                final_preds = avg_probs.argmax(axis=1)

        return ['benign', 'malignant'][int(final_preds[0])]

    def preprocess(self, image_path):

        image = Image.open(image_path).convert("RGB")
        transform = self.augmentations()
        processed_image = transform(image)

        return processed_image

    @staticmethod
    def augmentations():
        """
        Создает последовательность трансформаций для обработки изображения, включая ресайз, нормализацию и добавление шума.
        
        Возвращает:
            transforms.Compose: Композиция трансформаций.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        resize = transforms.Lambda(
            lambda img: resizeimage.resize_cover(img, [288, 288], validate=False)
        )

        add_gaussian_noise = transforms.Lambda(
            lambda img: img + torch.randn_like(img) * 0.02
        )

        to_tensor = transforms.ToTensor()

        val_transform = transforms.Compose([
            resize,
            to_tensor,
            add_gaussian_noise,
            normalize,
        ])

        return val_transform