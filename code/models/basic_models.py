# ************************************************************************
# * models/basic_models.py
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

# Definición de NET-1: Red neuronal sin capa oculta (regresión logística)
class NET1(nn.Module):
    def __init__(self, input_size=IMG_SIZE*IMG_SIZE):
        super(NET1, self).__init__()
        # Una red sin capas ocultas, directamente desde entrada a salida
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Definición de NET-2: Red neuronal con capas ocultas
class NET2(nn.Module):
    def __init__(self, input_size=IMG_SIZE*IMG_SIZE, hidden_sizes=None):
        super(NET2, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        self.flatten = nn.Flatten()
        
        # Construimos las capas ocultas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x
