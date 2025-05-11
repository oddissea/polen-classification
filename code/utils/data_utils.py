# utils/data_utils.py - Funciones de utilidad para carga, guardado y visualización
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import json

# Definimos parámetros generales que pueden ser usados en múltiples archivos
IMG_SIZE = 100  # Tamaño de las imágenes
BATCH_SIZE = 32
SEED = 42

# Clase personalizada para nuestro dataset de polen
class PolenDataset(Dataset):
    def __init__(self, images_paths, etiquetas=None, transform=None):
        """
        Dataset personalizado para imágenes de polen.
        
        Args:
            images_paths: Lista de rutas a las imágenes o directorio con imágenes
            etiquetas: Lista de etiquetas (opcional)
            transform: Transformaciones a aplicar a las imágenes
        """
        self.transform = transform
        self.images = []
        self.etiquetas = []
        
        # Si es un directorio, cargamos las imágenes y etiquetas
        if isinstance(images_paths, str) and os.path.isdir(images_paths):
            for archivo in os.listdir(images_paths):
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ruta_completa = os.path.join(images_paths, archivo)
                    self.images.append(ruta_completa)
                    # Asumimos que la etiqueta está en el nombre del directorio
                    if 'kunzea' in images_paths.lower():
                        self.etiquetas.append(0)  # Clase A (Kunzea) = 0
                    elif 'lepto' in images_paths.lower():
                        self.etiquetas.append(1)  # Clase B (Lepto) = 1
        # Si es una lista de rutas, las usamos directamente
        elif isinstance(images_paths, list):
            self.images = images_paths
            if etiquetas is not None:
                self.etiquetas = etiquetas
            else:
                # Si no se proporcionan etiquetas, intentamos inferirlas del nombre del directorio
                self.etiquetas = []
                for ruta in self.images:
                    if 'kunzea' in ruta.lower():
                        self.etiquetas.append(0)
                    elif 'lepto' in ruta.lower():
                        self.etiquetas.append(1)
                    else:
                        self.etiquetas.append(-1)  # Desconocido
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Cargamos la imagen como escala de grises
        imagen = Image.open(self.images[idx]).convert('L')
        etiqueta = self.etiquetas[idx]
        
        # Aplicamos las transformaciones si están definidas
        if self.transform:
            imagen = self.transform(imagen)
        
        return imagen, etiqueta

# Funciones para la gestión de datos

def guardar_datos_procesados(datos_info, directorio_salida='datos_procesados'):
    """
    Guarda la información de los datos procesados para uso posterior.
    
    Args:
        datos_info: Diccionario con información de los conjuntos de datos
        directorio_salida: Directorio donde guardar los datos
    """
    # Creamos el directorio si no existe
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Guardamos las rutas y etiquetas
    with open(os.path.join(directorio_salida, 'datos_procesados.pkl'), 'wb') as f:
        pickle.dump(datos_info, f) # type: ignore

    # Guardamos la configuración en formato JSON (más legible)
    config = {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'val_split': datos_info.get('val_split', 0.2),
        'transforms_config': datos_info.get('transforms_config', {}),
        'train_size': len(datos_info['train_images']),
        'val_size': len(datos_info['val_images']),
        'clases': {
            '0': 'Kunzea',
            '1': 'Lepto'
        }
    }
    
    with open(os.path.join(directorio_salida, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4) # type: ignore
    
    print(f"Datos procesados guardados en {directorio_salida}")

def cargar_datos_procesados(ruta_archivo='datos_procesados/datos_procesados.pkl'):
    """
    Carga los datos procesados previamente guardados.
    
    Args:
        ruta_archivo: Ruta al archivo de datos procesados
        
    Returns:
        Diccionario con información de los conjuntos de datos
    """
    with open(ruta_archivo, 'rb') as f:
        datos_info = pickle.load(f)
    
    return datos_info


def crear_dataloaders(datos_info, batch_size=32, transformaciones_aug=None, transformaciones_normal=None):
    """
    Crea los dataloaders para entrenamiento (con y sin augmentation), validación y test.

    Args:
        datos_info: Diccionario con los datos procesados
        batch_size: Tamaño del batch para los dataloaders
        transformaciones_aug: Transformaciones con data augmentation
        transformaciones_normal: Transformaciones normales sin data augmentation

    Returns:
        Tupla con los dataloaders (train_aug, train_normal, val, test)
    """
    if transformaciones_aug is None:
        transformaciones_aug = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.05),
            transforms.RandomAffine(0, translate=(0.03, 0.03)),
            transforms.ToTensor(),
        ])

    if transformaciones_normal is None:
        transformaciones_normal = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    # Creamos los datasets
    train_dataset_aug = PolenDataset(
        datos_info['train_images'],
        datos_info['train_etiquetas'],
        transform=transformaciones_aug
    )

    train_dataset_normal = PolenDataset(
        datos_info['train_images'],
        datos_info['train_etiquetas'],
        transform=transformaciones_normal
    )

    val_dataset = PolenDataset(
        datos_info['val_images'],
        datos_info['val_etiquetas'],
        transform=transformaciones_normal  # Sin data augmentation para validación
    )

    test_dataset = PolenDataset(
        datos_info['test_images'],
        datos_info['test_etiquetas'],
        transform=transformaciones_normal  # Sin data augmentation para test
    )

    # Creamos los dataloaders
    train_loader_aug = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True
    )

    train_loader_normal = DataLoader(
        train_dataset_normal,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader_aug, train_loader_normal, val_loader, test_loader

# Funciones para visualización

def visualizar_muestras(datos_info, num_muestras=5):
    """
    Visualiza muestras de imágenes de cada clase.
    
    Args:
        datos_info: Diccionario con información de los conjuntos de datos
        num_muestras: Número de muestras a mostrar por clase
    """
    # Separamos las imágenes por clase
    train_images_kunzea = [img for img, lbl in zip(datos_info['train_images'], datos_info['train_etiquetas']) if lbl == 0]
    train_images_lepto = [img for img, lbl in zip(datos_info['train_images'], datos_info['train_etiquetas']) if lbl == 1]
    
    # Seleccionamos muestras aleatorias
    muestras_kunzea = np.random.choice(train_images_kunzea, min(num_muestras, len(train_images_kunzea)), replace=False)
    muestras_lepto = np.random.choice(train_images_lepto, min(num_muestras, len(train_images_lepto)), replace=False)
    
    # Configuramos el gráfico
    fig, axes = plt.subplots(2, num_muestras, figsize=(15, 6))
    
    # Mostramos las muestras de Kunzea
    for i, ruta in enumerate(muestras_kunzea):
        imagen = Image.open(ruta).convert('L')
        axes[0, i].imshow(imagen, cmap='gray')
        axes[0, i].set_title(f"Kunzea (0)")
        axes[0, i].axis('off')
    
    # Mostramos las muestras de Lepto
    for i, ruta in enumerate(muestras_lepto):
        imagen = Image.open(ruta).convert('L')
        axes[1, i].imshow(imagen, cmap='gray')
        axes[1, i].set_title(f"Lepto (1)")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualizar_augmentation_detallada(ruta_imagen):
    """
    Visualiza ejemplos de data augmentation con títulos específicos para cada transformación.
    Esta función es SOLO para visualización y comprensión.
    
    Args:
        ruta_imagen: Ruta a la imagen a transformar
    """
    # Cargamos la imagen
    imagen_pil = Image.open(ruta_imagen).convert('L')
    
    # Verificamos el tamaño original
    ancho, alto = imagen_pil.size
    
    # Configuramos la semilla para reproducibilidad
    torch.manual_seed(SEED)
    
    # Definimos transformaciones individuales SOLO PARA VISUALIZACIÓN
    transform_original_tensor = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    transform_rotation = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(60),
        transforms.ToTensor()
    ])
    
    transform_brillo = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.05),
        transforms.ToTensor()
    ])
    
    transform_desplazamiento = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(0, translate=(0.03, 0.03)),
        transforms.ToTensor()
    ])
    
    # Definimos la transformación completa - igual a la usada en entrenamiento
    transform_completa = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(60),
        transforms.ColorJitter(brightness=0.05),
        transforms.RandomAffine(0, translate=(0.03, 0.03)),
        transforms.ToTensor()
    ])
    
    # Mostramos la imagen original y las transformaciones
    plt.figure(figsize=(15, 3))
    
    # Imagen original
    plt.subplot(1, 6, 1)
    plt.imshow(imagen_pil, cmap='gray')
    plt.title(f"Original\n{ancho}x{alto}px")
    plt.axis('off')
    
    # Imagen normalizada (redimensionada + convertida a tensor)
    imagen_normalizada = transform_original_tensor(imagen_pil)
    plt.subplot(1, 6, 2)
    plt.imshow(imagen_normalizada.squeeze(), cmap='gray')
    plt.title("Normalizada")
    plt.axis('off')
    
    # Rotación
    torch.manual_seed(SEED)
    imagen_rotada = transform_rotation(imagen_pil)
    plt.subplot(1, 6, 3)
    plt.imshow(imagen_rotada.squeeze(), cmap='gray')
    plt.title("Rotación 60°")
    plt.axis('off')
    
    # Brillo
    torch.manual_seed(SEED)
    imagen_brillo = transform_brillo(imagen_pil)
    plt.subplot(1, 6, 4)
    plt.imshow(imagen_brillo.squeeze(), cmap='gray')
    plt.title("Brillo ±5%")
    plt.axis('off')
    
    # Desplazamiento
    torch.manual_seed(SEED)
    imagen_desplazada = transform_desplazamiento(imagen_pil)
    plt.subplot(1, 6, 5)
    plt.imshow(imagen_desplazada.squeeze(), cmap='gray')
    plt.title("Desplaz. 3%")
    plt.axis('off')
    
    # Combinación completa (transformaciones_aug - las que se usan en entrenamiento)
    torch.manual_seed(SEED)
    imagen_combinada = transform_completa(imagen_pil)
    plt.subplot(1, 6, 6)
    plt.imshow(imagen_combinada.squeeze(), cmap='gray')
    plt.title("Combinación\ncompleta")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("NOTA: En el entrenamiento, solo se usa la transformación 'Combinación completa'")
    print("Las transformaciones individuales mostradas son solo para fines de visualización")

def visualizar_augmentation_variaciones(ruta_imagen, transform_aug=None, num_ejemplos=5):
    """
    Visualiza varias ejecuciones de la transformación completa que se usa en entrenamiento.
    
    Args:
        ruta_imagen: Ruta a la imagen a transformar
        transform_aug: Transformación de augmentation a aplicar (opcional)
        num_ejemplos: Número de variaciones a mostrar
    """
    # Cargamos la imagen
    imagen = Image.open(ruta_imagen).convert('L')
    
    # Si no se proporciona transformación, usamos la predeterminada
    if transform_aug is None:
        transform_aug = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.05),
            transforms.RandomAffine(0, translate=(0.03, 0.03)),
            transforms.ToTensor()
        ])
    
    # Mostramos la imagen original y varias ejecuciones de las transformaciones
    plt.figure(figsize=(15, 3))
    
    # Imagen original
    plt.subplot(1, num_ejemplos + 1, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Aplicamos la transformación completa varias veces con diferentes semillas
    for i in range(num_ejemplos):
        # Usamos una semilla diferente cada vez para obtener variaciones
        torch.manual_seed(SEED + i)
        
        # Aplicamos las transformaciones completas (las mismas que se usan en entrenamiento)
        imagen_aug = transform_aug(imagen)
        
        # Convertimos a numpy para visualización
        imagen_np = imagen_aug.numpy().squeeze()
        
        # Mostramos la imagen transformada
        plt.subplot(1, num_ejemplos + 1, i + 2)
        plt.imshow(imagen_np, cmap='gray')
        plt.title(f"Variación {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Variaciones aleatorias usando la transformación de entrenamiento", y=1.05)
    plt.show()
