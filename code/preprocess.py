# ************************************************************************
# * preprocess.py
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Minería de datos - PEC 3: Deep Learning
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************

import torch
import numpy as np
from torchvision import transforms

# Importamos funciones de utilidad
from utils.data_utils import (
    PolenDataset, 
    guardar_datos_procesados,
    visualizar_muestras,
    visualizar_augmentation_detallada,
    visualizar_augmentation_variaciones,
    crear_dataloaders,
    IMG_SIZE,
    SEED
)

# Configuración específica para el preprocesamiento
VAL_SPLIT =.2  # 20% para validación

# Configuramos semillas para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)

# Definimos las transformaciones para entrenamiento con data augmentation
transformaciones_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(60),  # Rotaciones de hasta 60 grados
    transforms.ColorJitter(brightness=0.05),  # Variaciones de brillo ±5%
    transforms.RandomAffine(0, translate=(0.03, 0.03)),  # Pequeños desplazamientos (3%)
    transforms.ToTensor(),  # Convierte a tensor y normaliza a [0, 1]
])

# Transformaciones sin data augmentation (solo redimensionar y convertir a tensor)
transformaciones_normal = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # Convierte a tensor y normaliza a [0, 1]
])


def cargar_y_dividir_datos(directorio_a, directorio_b):
    """
    Carga las imágenes, las combina y divide en conjuntos de entrenamiento, validación y test.

    Args:
        directorio_a: Ruta al directorio de imágenes de la clase A (Kunzea)
        directorio_b: Ruta al directorio de imágenes de la clase B (Lepto)

    Returns:
        Diccionario con información de los conjuntos de datos y rutas
    """
    print(f"Cargando imágenes de {directorio_a}...")
    dataset_a = PolenDataset(directorio_a, transform=None)

    print(f"Cargando imágenes de {directorio_b}...")
    dataset_b = PolenDataset(directorio_b, transform=None)

    # Combinamos las rutas y etiquetas
    todas_images = dataset_a.images + dataset_b.images
    todas_etiquetas = dataset_a.etiquetas + dataset_b.etiquetas

    # Creamos índices y barajamos
    indices = list(range(len(todas_images)))
    np.random.shuffle(indices)

    # Calculamos tamaños de los conjuntos (70% train, 15% validación, 15% test)
    n_test = int(len(indices) * 0.15)
    n_val = int(len(indices) * 0.15)
    n_train = len(indices) - n_val - n_test

    # Dividimos los índices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Extraemos los datos para cada conjunto
    train_images = [todas_images[i] for i in train_indices]
    train_etiquetas = [todas_etiquetas[i] for i in train_indices]

    val_images = [todas_images[i] for i in val_indices]
    val_etiquetas = [todas_etiquetas[i] for i in val_indices]

    test_images = [todas_images[i] for i in test_indices]
    test_etiquetas = [todas_etiquetas[i] for i in test_indices]

    # Imprimimos estadísticas
    print(f"Total de imágenes: {len(todas_images)}")
    print(f"Distribución de clases: {todas_etiquetas.count(0)} Kunzea, {todas_etiquetas.count(1)} Lepto")
    print(f"Conjunto de entrenamiento: {len(train_images)} imágenes")
    print(f"Conjunto de validación: {len(val_images)} imágenes")
    print(f"Conjunto de test: {len(test_images)} imágenes")

    # Preparamos la información para guardar
    datos_info = {
        'train_images': train_images,
        'train_etiquetas': train_etiquetas,
        'val_images': val_images,
        'val_etiquetas': val_etiquetas,
        'test_images': test_images,
        'test_etiquetas': test_etiquetas,
        'splits': {
            'train': len(train_images) / len(todas_images),
            'val': len(val_images) / len(todas_images),
            'test': len(test_images) / len(todas_images)
        },
        'transforms_config': {
            'img_size': IMG_SIZE,
            'rotation_range': 60,
            'brightness_range': 0.05,
            'shift_range': 0.03,
        }
    }

    return datos_info


def ejecutar_preprocesamiento(directorio_a, directorio_b, dir_salida='datos_procesados'):
    """
    Ejecuta el flujo completo de preprocesamiento y guardado.

    Args:
        directorio_a: Ruta al directorio de imágenes de la clase A (Kunzea)
        directorio_b: Ruta al directorio de imágenes de la clase B (Lepto)
        dir_salida: Directorio donde guardar los datos procesados
    """
    print("Iniciando preprocesamiento de datos...")

    # Cargamos y dividimos los datos
    datos_info = cargar_y_dividir_datos(directorio_a, directorio_b)

    # Guardamos los datos procesados
    guardar_datos_procesados(datos_info, dir_salida)

    # Visualizamos algunas muestras
    print("\nVisualizando muestras de cada clase:")
    visualizar_muestras(datos_info)

    # Seleccionamos una imagen de ejemplo de cada clase
    imagen_kunzea = datos_info['train_images'][datos_info['train_etiquetas'].index(0)]
    imagen_lepto = datos_info['train_images'][datos_info['train_etiquetas'].index(1)]

    # Visualizamos transformaciones detalladas
    print("\nVisualizando transformaciones detalladas para Kunzea:")
    visualizar_augmentation_detallada(imagen_kunzea)

    # Visualizamos varias ejecuciones aleatorias de la transformación completa
    print("\nVisualizando variaciones aleatorias del data augmentation para Kunzea:")
    visualizar_augmentation_variaciones(imagen_kunzea, transformaciones_aug)

    print("\nVisualizando transformaciones detalladas para Lepto:")
    visualizar_augmentation_detallada(imagen_lepto)

    # Visualizamos varias ejecuciones aleatorias de la transformación completa
    print("\nVisualizando variaciones aleatorias del data augmentation para Lepto:")
    visualizar_augmentation_variaciones(imagen_lepto, transformaciones_aug)

    # Creamos los dataloaders (solo para mostrar que funciona)
    train_loader_aug, train_loader_normal, val_loader, test_loader = crear_dataloaders(
        datos_info,
        transformaciones_aug=transformaciones_aug,
        transformaciones_normal=transformaciones_normal
    )

    print("\nPreprocesamiento completado. Datos guardados en:", dir_salida)
    print("Puedes cargar estos datos en otros notebooks utilizando la función cargar_datos_procesados()")

    return datos_info, (train_loader_aug, train_loader_normal, val_loader, test_loader)

# Ejemplo de uso
if __name__ == "__main__":
    # Rutas a los directorios de imágenes (ajustar según la estructura real)
    TIPO_A_DIR = 'anuka1200/Tipo A: Kunzea'
    TIPO_B_DIR = 'anuka1200/Tipo B: Lepto'
    
    # Ejecutamos el preprocesamiento
    datos, loaders = ejecutar_preprocesamiento(TIPO_A_DIR, TIPO_B_DIR)
