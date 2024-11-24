from torchensemble import VotingClassifier
from torchensemble.utils import io
from models.WideResNet import ResNet
import torch

def main():
    """
    Создает ансамбль классификаторов на основе VotingClassifier с архитектурой ResNet и загружает предварительно обученные веса.

    Возвращает:
        list: Список, содержащий ансамбль классификаторов VotingClassifier.
    """
    ensemble = VotingClassifier(
        estimator=ResNet,                     # Базовый классификатор
        estimator_args={"outputs_number": 2}, # Параметры классификатора (2 класса)
        n_estimators=4,                       # Количество моделей в ансамбле
        cuda=False                            # Использовать CPU
    )

    weights_dir = 'weights'
    io.load(ensemble, weights_dir, map_location=torch.device('cpu'))
    return [ensemble]
