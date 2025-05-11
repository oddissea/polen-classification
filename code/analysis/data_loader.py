# ************************************************************************
# * analysis/data_loader.py
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Minería de datos - PEC 3: Deep Learning
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************
"""
Módulo para cargar datos de entrenamiento, modelos y métricas guardadas.
"""
import os
import pickle
import torch
import numpy as np
from models.basic_models import NET1, NET2
from models.cnn_models import CNNBase, CNNVariant1, CNNVariant2
from utils.data_utils import cargar_datos_procesados, crear_dataloaders, IMG_SIZE

# Configuración de semillas para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def get_device():
    """
    Determina el dispositivo disponible para cálculos (MPS, CUDA o CPU).

    Returns:
        torch.device: Dispositivo a utilizar para entrenamiento
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_data(data_path='datos_procesados/datos_procesados.pkl'):
    """
    Carga los datos procesados y crea los dataloaders.

    Args:
        data_path: Ruta al archivo de datos procesados

    Returns:
        tuple: Tuple con (train_loader_aug, train_loader_normal, val_loader, test_loader)
    """
    # Cargamos los datos procesados
    print("Cargando datos procesados...")
    datos_info = cargar_datos_procesados(data_path)

    # Creamos los dataloaders
    train_loader_aug, train_loader_normal, val_loader, test_loader = crear_dataloaders(datos_info)

    print(f"Total de imágenes de entrenamiento: {len(train_loader_aug.dataset)}")
    print(f"Total de imágenes de validación: {len(val_loader.dataset)}")
    print(f"Total de imágenes de test: {len(test_loader.dataset)}")

    return train_loader_aug, train_loader_normal, val_loader, test_loader


def load_training_histories(results_dir='resultados'):
    """
    Carga los historiales de entrenamiento desde archivos pickle.

    Args:
        results_dir: Directorio donde se encuentran los archivos de resultados

    Returns:
        dict: Diccionario combinado con todos los historiales
    """
    historiales_basicos = {}
    historiales_cnn = {}

    # Intentamos cargar historiales de modelos básicos
    try:
        with open(os.path.join(results_dir, 'historiales_modelos_basicos.pkl'), 'rb') as f:
            historiales_basicos = pickle.load(f)
        print("Historiales de modelos básicos cargados correctamente")

        # Procesamos los historiales para asegurar consistencia con el nuevo formato
        for key, historial in historiales_basicos.items():
            # Si no existe 'early_stopping_epoch', lo inferimos de 'stopped_epoch' si existe
            if 'early_stopping_epoch' not in historial:
                if 'stopped_epoch' in historial:
                    historial['early_stopping_epoch'] = historial['stopped_epoch']
                else:
                    # Si no tenemos 'stopped_epoch', asumimos que terminó en la última época registrada
                    historial['early_stopping_epoch'] = len(historial['train_acc'])

            # Si no existe 'finished_epoch', lo inferimos de la longitud de las métricas
            if 'finished_epoch' not in historial:
                historial['finished_epoch'] = len(historial['train_acc'])

    except Exception as e:
        print(f"Error al cargar historiales básicos: {e}")

    # Intentamos cargar historiales de modelos CNN
    try:
        with open(os.path.join(results_dir, 'historiales_modelos_cnn.pkl'), 'rb') as f:
            historiales_cnn = pickle.load(f)
        print("Historiales de modelos CNN cargados correctamente")

        # Procesamos los historiales CNN de manera similar
        for key, historial in historiales_cnn.items():
            if 'early_stopping_epoch' not in historial:
                if 'stopped_epoch' in historial:
                    historial['early_stopping_epoch'] = historial['stopped_epoch']
                else:
                    historial['early_stopping_epoch'] = len(historial['train_acc'])

            if 'finished_epoch' not in historial:
                historial['finished_epoch'] = len(historial['train_acc'])
    except Exception as e:
        print(f"Error al cargar historiales CNN: {e}")

    # Combinamos todos los historiales
    return {**historiales_basicos, **historiales_cnn}


def load_test_metrics(results_dir='resultados', load_final=False):
    """
    Carga las métricas de test desde archivos pickle.

    Args:
        results_dir: Directorio donde se encuentran los archivos de resultados
        load_final: Si True, carga las métricas de modelos finales en lugar de modelos optimizados

    Returns:
        dict: Diccionario combinado con todas las métricas de test
    """
    test_metrics_basicos = {}
    test_metrics_cnn = {}

    # Determinamos el sufijo de los archivos según el tipo a cargar
    suffix = "_final" if load_final else ""

    # Intentamos cargar métricas de test para modelos básicos
    try:
        with open(os.path.join(results_dir, f'test_metrics_basicos{suffix}.pkl'), 'rb') as f:
            test_metrics_basicos = pickle.load(f)
        print(f"Métricas de test de modelos básicos{' finales' if load_final else ''} cargadas correctamente")
    except Exception as e:
        print(f"Error al cargar métricas de test básicos{' finales' if load_final else ''}: {e}")

    # Intentamos cargar métricas de test para modelos CNN
    try:
        with open(os.path.join(results_dir, f'test_metrics_cnn{suffix}.pkl'), 'rb') as f:
            test_metrics_cnn = pickle.load(f)
        print(f"Métricas de test de modelos CNN{' finales' if load_final else ''} cargadas correctamente")
    except Exception as e:
        print(f"Error al cargar métricas de test CNN{' finales' if load_final else ''}: {e}")

    # Combinamos todas las métricas
    return {**test_metrics_basicos, **test_metrics_cnn}


def cargar_modelo(modelo_clase, ruta_pesos, nombre="model", device=None):
    """
    Crea una instancia de un model y carga sus pesos.

    Args:
        modelo_clase: Clase del model (NET1, NET2, CNNBase, etc.)
        ruta_pesos: Ruta al archivo de pesos
        nombre: Nombre descriptivo del model
        device: Dispositivo donde cargar el modelo (si es None, se determina automáticamente)

    Returns:
        El model con los pesos cargados o None si hay error
    """
    if device is None:
        device = get_device()

    try:
        if modelo_clase in [NET1, NET2]:
            model = modelo_clase(input_size=IMG_SIZE * IMG_SIZE)
        else:
            model = modelo_clase()

        # Agregamos weights_only=True para evitar el warning de seguridad
        model.load_state_dict(torch.load(ruta_pesos, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"Modelo {nombre} cargado correctamente")
        return model
    except Exception as ex:
        print(f"Error al cargar el model {nombre}: {ex}")
        return None


def load_models(results_dir='resultados', load_best=True):
    """
    Carga todos los modelos entrenados.

    Args:
        results_dir: Directorio donde se encuentran los modelos guardados
        load_best: Si True, carga los modelos guardados con early stopping (best),
                  si False, carga los modelos finales

    Returns:
        dict: Diccionario con todos los modelos cargados
    """
    device = get_device()
    print(f'Utilizando dispositivo: {device}')

    # Sufijo para los archivos de pesos según el tipo de modelo a cargar
    suffix = "_best.pth" if load_best else "_final.pth"

    # Creamos un diccionario para almacenar todos los modelos
    todos_modelos = {}

    # Cargamos los modelos básicos (NET1 y NET2)
    print(f"Cargando modelos básicos ({'mejores' if load_best else 'finales'})...")

    # NET1
    net1_normal = cargar_modelo(
        NET1,
        os.path.join(results_dir, f'NET1_normal{suffix}'),
        f'NET1 Normal {"(best)" if load_best else "(final)"}',
        device
    )
    if net1_normal:
        todos_modelos['NET1_normal'] = net1_normal

    net1_aug = cargar_modelo(
        NET1,
        os.path.join(results_dir, f'NET1_aug{suffix}'),
        f'NET1 Augmented {"(best)" if load_best else "(final)"}',
        device
    )
    if net1_aug:
        todos_modelos['NET1_aug'] = net1_aug

    # NET2
    net2_normal = cargar_modelo(
        NET2,
        os.path.join(results_dir, f'NET2_normal{suffix}'),
        f'NET2 Normal {"(best)" if load_best else "(final)"}',
        device
    )
    if net2_normal:
        todos_modelos['NET2_normal'] = net2_normal

    net2_aug = cargar_modelo(
        NET2,
        os.path.join(results_dir, f'NET2_aug{suffix}'),
        f'NET2 Augmented {"(best)" if load_best else "(final)"}',
        device
    )
    if net2_aug:
        todos_modelos['NET2_aug'] = net2_aug

    # Cargamos los modelos CNN
    print(f"Cargando modelos CNN ({'mejores' if load_best else 'finales'})...")

    # CNN Base
    cnn_base_normal = cargar_modelo(
        CNNBase,
        os.path.join(results_dir, f'CNN_base_normal{suffix}'),
        f'CNN Base (normal) {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_base_normal:
        todos_modelos['CNN_base_normal'] = cnn_base_normal

    cnn_base_aug = cargar_modelo(
        CNNBase,
        os.path.join(results_dir, f'CNN_base_aug{suffix}'),
        f'CNN Base {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_base_aug:
        todos_modelos['CNN_base_aug'] = cnn_base_aug

    # CNN Variante 1
    cnn_var1_normal = cargar_modelo(
        CNNVariant1,
        os.path.join(results_dir, f'CNN_variant1_normal{suffix}'),
        f'CNN Variante 1 (normal) {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_var1_normal:
        todos_modelos['CNN_variant1_normal'] = cnn_var1_normal

    cnn_var1_aug = cargar_modelo(
        CNNVariant1,
        os.path.join(results_dir, f'CNN_variant1_aug{suffix}'),
        f'CNN Variante 1 {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_var1_aug:
        todos_modelos['CNN_variant1_aug'] = cnn_var1_aug

    # CNN Variante 2
    cnn_var2_normal = cargar_modelo(
        CNNVariant2,
        os.path.join(results_dir, f'CNN_variant2_normal{suffix}'),
        f'CNN Variante 2 (normal) {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_var2_normal:
        todos_modelos['CNN_variant2_normal'] = cnn_var2_normal

    cnn_var2_aug = cargar_modelo(
        CNNVariant2,
        os.path.join(results_dir, f'CNN_variant2_aug{suffix}'),
        f'CNN Variante 2 {"(best)" if load_best else "(final)"}',
        device
    )
    if cnn_var2_aug:
        todos_modelos['CNN_variant2_aug'] = cnn_var2_aug

    return todos_modelos


def obtener_imagen_muestra(dataloader, clase=0):
    """
    Obtiene una imagen de muestra de la clase especificada.

    Args:
        dataloader: DataLoader del que obtener la imagen
        clase: Clase deseada (0=Kunzea, 1=Lepto)

    Returns:
        Tuple: (imagen_tensor, imagen_np)
    """

    for images, labels in dataloader:
        # Buscamos una imagen de la clase deseada
        for index, label in enumerate(labels):
            if label == clase:
                imagen = images[index:index + 1]  # Incluimos la dimensión de batch
                return imagen, imagen.squeeze().cpu().numpy()

    print(f"No se encontró ninguna imagen de la clase {clase}")
    return None, None


def liberar_memoria():
    """Libera memoria de GPU/MPS después del análisis"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    print("Memoria liberada")