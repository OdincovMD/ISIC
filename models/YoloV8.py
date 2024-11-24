from ultralytics import YOLO
import os

def main():
    """
    Загружает YOLO-модель с предобученными весами из указанной директории.

    Возвращает:
        list: Список, содержащий загруженную модель YOLO.

    Исключения:
        FileNotFoundError: Если файл весов не найден.
    """
    weights_path = os.path.join('weights', 'best.pt')
    model = YOLO(weights_path)
    return [model]
