import torch
import os

class ResNet(torch.nn.Module):
    """
    Класс для модифицированной архитектуры WideResNet50_2 с замороженными слоями и добавленным выходным слоем.

    Аргументы:
        outputs_number (int): Количество выходных классов.
    """
    def __init__(self, outputs_number):
        super(ResNet, self).__init__()
        # Загрузка предобученной модели WideResNet50_2
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)

        # Заморозка всех параметров модели
        for param in self.net.parameters():
            param.requires_grad = False

        # Разморозка параметров последнего блока
        for param in self.net.layer4.parameters():
            param.requires_grad = True

        # Настройка выходного слоя
        TransferModelOutputs = self.net.fc.in_features  # Размерность входа в последний слой
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, 512),  # Промежуточный слой
            torch.nn.ReLU(),  # Нелинейная активация
            torch.nn.BatchNorm1d(512),  # Нормализация
            torch.nn.Dropout(0.6),  # Дроп-аут для регуляризации
            torch.nn.Linear(512, outputs_number)  # Выходной слой
        )

    def forward(self, x):
        return self.net(x)

def main():
    """
    Загружает ансамбль из 4 моделей с предобученными весами.

    Возвращает:
        list: Список моделей ResNet.
    """
    models = []
    for fold in range(4):
        model = ResNet(2)
        weight_path = os.path.join('weights', f'best_model_fold_{fold+1}.pth')
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        models.append(model)

    return models
