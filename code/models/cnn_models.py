# ************************************************************************
# * models/cnn_models.py
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Minería de datos - PEC 3: Deep Learning
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************
import torch.nn as nn

# Importamos el tamaño de imagen de data_utils
from utils.data_utils import IMG_SIZE

# Definición de del model base CNN (Modelo 3)
class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()
        
        # Primera secuencia convolucional
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Segunda secuencia convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Tercera secuencia convolucional
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculamos el tamaño después de las convoluciones y pooling
        # Después de 3 capas de MaxPool con stride=2, el tamaño se reduce por 2^3 = 8
        conv_output_size = IMG_SIZE // 8
        flattened_size = 64 * conv_output_size * conv_output_size
        
        # Capas completamente conectadas
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

# Definición de la Variante 1 de CNN (Modelo 4) - Usando BatchNorm y más filtros
class CNNVariant1(nn.Module):
    def __init__(self):
        super(CNNVariant1, self).__init__()
        
        # Primera secuencia convolucional con BatchNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Segunda secuencia convolucional con BatchNorm
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Tercera secuencia convolucional con BatchNorm
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculamos el tamaño después de las convoluciones y pooling
        conv_output_size = IMG_SIZE // 8
        flattened_size = 128 * conv_output_size * conv_output_size
        
        # Capas completamente conectadas
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

# Definición de la Variante 2 de CNN (Modelo 5) - Usando diferentes tamaños de kernel
class CNNVariant2(nn.Module):
    def __init__(self):
        super(CNNVariant2, self).__init__()
        
        # Primera secuencia convolucional con kernel 5x5
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Segunda secuencia convolucional con kernel 3x3
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Tercera secuencia convolucional con stride=1 y sin pooling
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Cuarta secuencia convolucional con kernel 1x1 (ajustando canales)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculamos el tamaño después de las convoluciones y pooling
        # Después de 3 capas de MaxPool con stride=2, el tamaño se reduce por 2^3 = 8
        conv_output_size = IMG_SIZE // 8
        flattened_size = 128 * conv_output_size * conv_output_size
        
        # Capas completamente conectadas
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x
