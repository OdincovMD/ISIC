import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(models: list, dataloader, device: str, voting : str='soft') -> tuple:
    """
    Функция для предсказания и оценки качества модели с использованием ансамбля моделей.

    Аргументы:
        models (list): Список обученных моделей. Последняя модель используется для обработки путей (path).
        dataloader (torch.utils.data.DataLoader): Даталоадер для тестовых данных.
        device (torch.device): Устройство для вычислений (CPU/GPU).
        voting (str): Метод голосования. Возможные значения:
                      - 'soft': Soft voting (усреднение вероятностей).
                      - 'majority': Majority voting (мода предсказаний).

    Возвращает:
        tuple: Метрики модели (accuracy, precision, recall, f1).
    """
    all_labels = []
    all_preds = []

    # Перевод всех моделей в режим оценки
    for model in models[:-1]:
        model.eval()

    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader, desc="Processing data"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            fold_preds = []
            fold_probs = []

            for i, model in enumerate(models[:-1]):
                if i != 4:  # Предполагается, что 4-я модель имеет другой интерфейс
                    preds = model(inputs)  # Предсказания модели
                    probs = torch.softmax(preds, dim=1)  # Вероятности классов
                else:
                    probs = model.predict(inputs)  # Предполагается, что модель возвращает вероятности напрямую

                fold_probs.append(probs.cpu().numpy())  # Для soft voting
                preds_class = probs.argmax(dim=1)  # Определение класса
                fold_preds.append(preds_class.cpu().numpy())

            # Обработка последней модели с использованием путей
            result = models[-1](paths[0], verbose=False)
            probs = result[0].probs.data.unsqueeze(0)
            fold_probs.append(probs.cpu().numpy())
            preds_class = probs.argmax(dim=1)
            fold_preds.append(preds_class.cpu().numpy())

            # Голосование
            if voting == 'majority':
                # Majority voting (мода по всем предсказаниям)
                preds_voted = np.stack(fold_preds, axis=1)  # Сохраняем предсказания всех моделей
                final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds_voted)

            elif voting == 'soft':
                # Soft voting (усреднение вероятностей)
                avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
                final_preds = avg_probs.argmax(axis=1)

            all_preds.extend(final_preds)
            all_labels.extend(labels.cpu().numpy())

    # Вычисление метрик
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall, f1
