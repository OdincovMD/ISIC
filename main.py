import models.WideResNet as WideResNet
import models.VotingClassifier as VotingClassifier
import models.YoloV8 as YoloV8
from inference import data, predict

def main(path: str, device: str = 'cpu', voting: str = 'soft'):
    """
    Основная функция для выполнения предсказания с использованием нескольких моделей и вывода метрик.

    Аргументы:
        path (str): Путь к данным (например, 'test').
        device (str, optional): Устройство для выполнения ('cpu' или 'cuda'). По умолчанию 'cpu'.
        voting (str, optional): Метод голосования ('soft' или 'majority'). По умолчанию 'soft'.

    Возвращает:
        None
    """
    try:
        # Инициализация моделей
        print("Загрузка моделей...")
        models = WideResNet.main() \
            + VotingClassifier.main() \
            + YoloV8.main()
        
        print(f"Загружено {len(models)} моделей.")

        # Загрузка данных
        print("Загрузка данных...")
        dataloader = data.main(path)

        # Выполнение предсказания
        print("Выполнение предсказания...")
        accuracy, precision, recall, f1 = predict.main(models, dataloader, device, voting)

        # Вывод метрик
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except RuntimeError as e:
        print(f"Ошибка выполнения: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

if __name__ == '__main__':
    main(path='test')
