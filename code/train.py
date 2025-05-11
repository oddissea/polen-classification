# ************************************************************************
# * train.py
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Minería de datos - PEC 3: Deep Learning
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************

# Librerías necesarias
import os
import gc
import numpy as np
import pickle as pkl
import time
from datetime import datetime

# Añade estas importaciones al principio
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


import torch
import torch.nn as nn
import torch.optim as optim


# Importamos los modelos
from models.cnn_models import CNNBase, CNNVariant1, CNNVariant2
from models.basic_models import NET1, NET2
# Importamos las funciones de utilidad
from utils.data_utils import cargar_datos_procesados, crear_dataloaders, IMG_SIZE


# Configuración de semillas para reproducibilidad
def config_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Configuramos dispositivo para entrenamiento
def config_device():
    if torch.backends.mps.is_available():
        d = torch.device("mps")
    elif torch.cuda.is_available():
        d = torch.device("cuda")
    else:
        d = torch.device("cpu")
    return d

def liberar_memoria():
    """Libera memoria de GPU/MPS después del entrenamiento"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    print("Memoria liberada")

def evaluate_model(model, test_data, criterion, model_name):
    """
    Evalúa el model en el conjunto de test.

    Args:
        model: Modelo entrenado
        test_data: DataLoader del conjunto de test
        criterion: Función de pérdida
        model_name: Nombre del model para guardar resultados

    Returns:
        dict: Métricas de evaluación
    """

    # Asegurarse de que el model esté en el mismo dispositivo que los datos
    model = model.to(device)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Para calcular la matriz de confusión
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item() # type: ignore

            # Guardamos predicciones y etiquetas para la matriz de confusión
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    # Calculamos la matriz de confusión
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculamos precision, recall y f1-score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n--- Evaluación en test del model {model_name} ---")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Matriz de confusión:")
    print(conf_matrix)

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix
    }

# Función de entrenamiento y validación
def train(model, train_data, val_data, criterion, optimizer, num_epochs, model_name, patience=10):
    """
    Entrena el model y evalúa su rendimiento.

    Args:
        model: Modelo a entrenar
        train_data: DataLoader de entrenamiento
        val_data: DataLoader de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        num_epochs: Número de épocas
        model_name: Nombre del model para guardar resultados
        patience: Número de épocas a esperar sin mejora antes de detener el entrenamiento

    Returns:
        dict: Historial de métricas durante el entrenamiento
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0
    early_stopping_epoch = None  # Nuevo: guardar en qué época se activaría el early stopping

    # Inicializamos epoch fuera del bucle
    epoch = 0

    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # Ajustamos para BCELoss

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Actualizar estadísticas
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item() # type: ignore

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Modo evaluación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_data:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item() # type: ignore

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Guardamos métricas
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Guardamos el mejor model (checkpoint)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model_name}_best.pth"))
        else:
            epochs_without_improvement += 1

        # Informamos del progreso
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Detectamos early stopping pero continuamos entrenando
        if epochs_without_improvement >= patience and early_stopping_epoch is None:
            early_stopping_epoch = epoch + 1
            print(f"Early stopping detectado en época {early_stopping_epoch}, pero continuando entrenamiento")

    # Guardamos el model final
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model_name}_final.pth"))

    # Añadimos la época de early stopping al historial
    history['early_stopping_epoch'] = early_stopping_epoch if early_stopping_epoch else epoch + 1
    history['finished_epoch'] = epoch + 1

    # Al final del entrenamiento, antes de retornar
    print(f"Mejor precisión de validación: {best_val_acc:.4f}")
    print(f"Early stopping detectado en época: {early_stopping_epoch if early_stopping_epoch else 'No detectado'}")
    print(f"Entrenamiento completado en la época {epoch + 1}")

    return history

def train_model(model_class, train_data, model_name, lr, wd, size=None, patience=10):
    net1 = model_class(size) if size is not None else model_class()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net1.parameters(), lr=lr, weight_decay=wd)
    inicio_net1_normal = time.time()
    history = train(
        net1,
        train_data,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        model_name,
        patience=patience
    )
    training_time = time.time() - inicio_net1_normal
    history['training_time'] = training_time
    return history


def print_test_summary(test_results_dict, title, is_final=False):
    """
    Imprime un resumen de los resultados de test para todos los modelos.

    Args:
        test_results_dict: Diccionario con los resultados de test
        title: Título para el resumen
        is_final: Si es True, indica que son modelos finales (sin early stopping)
    """
    model_type = "finales" if is_final else "optimizados"
    print(f"\n=== {title} ({model_type}) ===")

    # Sin data augmentation
    print("Sin data augmentation:")
    print(f"NET1: {test_results_dict['NET1_normal']['test_acc']:.4f}")
    print(f"NET2: {test_results_dict['NET2_normal']['test_acc']:.4f}")
    print(f"CNN Base: {test_results_dict['CNN_base_normal']['test_acc']:.4f}")
    print(f"CNN Variante 1: {test_results_dict['CNN_variant1_normal']['test_acc']:.4f}")
    print(f"CNN Variante 2: {test_results_dict['CNN_variant2_normal']['test_acc']:.4f}")

    # Con data augmentation
    print("\nCon data augmentation:")
    print(f"NET1: {test_results_dict['NET1_aug']['test_acc']:.4f}")
    print(f"NET2: {test_results_dict['NET2_aug']['test_acc']:.4f}")
    print(f"CNN Base: {test_results_dict['CNN_base_aug']['test_acc']:.4f}")
    print(f"CNN Variante 1: {test_results_dict['CNN_variant1_aug']['test_acc']:.4f}")
    print(f"CNN Variante 2: {test_results_dict['CNN_variant2_aug']['test_acc']:.4f}")

    # Mejora por data augmentation
    print(f"\nMejora debido al data augmentation (en test, modelos {model_type}):")
    print(
        f"NET1: {(test_results_dict['NET1_aug']['test_acc'] - test_results_dict['NET1_normal']['test_acc']) * 100:.2f}%")
    print(
        f"NET2: {(test_results_dict['NET2_aug']['test_acc'] - test_results_dict['NET2_normal']['test_acc']) * 100:.2f}%")
    print(
        f"CNN Base: {(test_results_dict['CNN_base_aug']['test_acc'] - test_results_dict['CNN_base_normal']['test_acc']) * 100:.2f}%")
    print(
        f"CNN Variante 1: {(test_results_dict['CNN_variant1_aug']['test_acc'] - test_results_dict['CNN_variant1_normal']['test_acc']) * 100:.2f}%")
    print(
        f"CNN Variante 2: {(test_results_dict['CNN_variant2_aug']['test_acc'] - test_results_dict['CNN_variant2_normal']['test_acc']) * 100:.2f}%")


if __name__ == '__main__':
    # Semilla de reproducibilidad
    SEED = 42

    config_seed(SEED)
    device = config_device()
    print(f'Utilizando dispositivo: {device}')

    # Parámetros de entrenamiento
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LR = [1e-2, 1e-3, 5e-4, 1e-3, 3e-4]
    PATIENCE = [5, 7, 10, 12, 15]
    WEIGHT_DECAY = 1e-5  # Regularización L2

    # Directorio para guardar los resultados
    RESULTS_DIR = 'resultados'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cargamos los datos previamente procesados
    print("Cargando datos procesados...")
    datos_info = cargar_datos_procesados('datos_procesados/datos_procesados.pkl')

    # Creamos los dataloaders (con y sin data augmentation)
    train_loader_aug, train_loader_normal, val_loader, test_loader = crear_dataloaders(
        datos_info,
        batch_size=BATCH_SIZE
    )

    print(f"Total de imágenes de entrenamiento: {len(train_loader_aug.dataset)}")
    print(f"Total de imágenes de validación: {len(val_loader.dataset)}")

    input_size = IMG_SIZE * IMG_SIZE

    # Análisis de tiempo de ejecución
    print(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Entrenamos NET-1 (sin augmentation)
    print("\n--- Entrenando NET-1 (sin augmentation) ---")
    history_net1_normal = train_model(NET1, train_loader_normal,
                                      "NET1_normal",
                                      lr=LR[0],
                                      wd=WEIGHT_DECAY,
                                      size=input_size, patience=PATIENCE[0])
    print(f"Tiempo de entrenamiento NET-1 (sin augmentation): {history_net1_normal['training_time']:.2f} segundos")

    # Entrenamos NET-1 (con augmentation)
    print("\n--- Entrenando NET-1 (con augmentation) ---")
    history_net1_aug = train_model(NET1, train_loader_aug,
                                   "NET1_aug",
                                   lr=LR[0], wd=WEIGHT_DECAY,
                                   size=input_size, patience=PATIENCE[0])
    print(f"Tiempo de entrenamiento NET-1 (con augmentation): {history_net1_aug['training_time']:.2f} segundos")


    # Entrenamos NET-2 (sin augmentation)
    print("\n--- Entrenando NET-2 (sin augmentation) ---")
    history_net2_normal = train_model(NET2, train_loader_normal,
                                      "NET2_normal",
                                      lr=LR[1], wd=WEIGHT_DECAY,
                                      size=input_size, patience=PATIENCE[1])
    print(f"Tiempo de entrenamiento NET-2 (sin augmentation): {history_net2_normal['training_time']:.2f} segundos")

    # Entrenamos NET-2 (con augmentation)
    print("\n--- Entrenando NET-2 (con augmentation) ---")
    history_net2_aug = train_model(NET2, train_loader_aug,
                                   "NET2_aug",
                                   lr=LR[1], wd=WEIGHT_DECAY,
                                   size=input_size, patience=PATIENCE[1])
    print(f"Tiempo de entrenamiento NET-2 (con augmentation): {history_net2_aug['training_time']:.2f} segundos")

    print(f"Fecha y hora de finalización del entrenamiento de los modelos normales: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Guardando...\n")
    with open(os.path.join(RESULTS_DIR, 'historiales_modelos_basicos.pkl'), 'wb') as f:
        pkl.dump({
            'NET1_normal': history_net1_normal,
            'NET1_aug': history_net1_aug,
            'NET2_normal': history_net2_normal,
            'NET2_aug': history_net2_aug
        }, f) # type: ignore

    print("Entrenamiento de modelos básicos completado.")
    print(f"Resultados guardados en {RESULTS_DIR}/")

    # Análisis de tiempo de ejecución
    print(f"Fecha y hora de inicio CNN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Resumen de los mejores resultados
    print("\n=== Resumen de resultados ===")
    print(f"NET1 (normal): Mejor exactitud de validación = {max(history_net1_normal['val_acc']):.4f}")
    print(f"NET1 (aug): Mejor exactitud de validación = {max(history_net1_aug['val_acc']):.4f}")
    print(f"NET2 (normal): Mejor exactitud de validación = {max(history_net2_normal['val_acc']):.4f}")
    print(f"NET2 (aug): Mejor exactitud de validación = {max(history_net2_aug['val_acc']):.4f}")

    # Entrenamos la CNN base (Modelo 3) sin data augmentation
    print("\n--- Entrenando CNN Base (Modelo 3) sin data augmentation ---")
    history_cnn_base_normal = train_model(CNNBase, train_loader_normal,
                                          "CNN_base_normal",
                                          lr=LR[2],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[2])
    print(f"Tiempo de entrenamiento CNN Base (Modelo 3) sin data augmentation: "
          f"{history_cnn_base_normal['training_time']:.2f} segundos")

    # Entrenamos la CNN base (Modelo 3) con data augmentation
    print("\n--- Entrenando CNN Base (Modelo 3) con data augmentation ---")
    history_cnn_base_aug = train_model(CNNBase, train_loader_aug,
                                          "CNN_base_aug",
                                          lr=LR[2],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[2])
    print(f"Tiempo de entrenamiento CNN Base (Modelo 3) con data augmentation: "
          f"{history_cnn_base_aug['training_time']:.2f} segundos")

    # Entrenamos la Variante 1 (Modelo 4) sin data augmentation
    print("\n--- Entrenando CNN Variante 1 (Modelo 4) sin data augmentation ---")
    history_cnn_var1_normal = train_model(CNNVariant1, train_loader_normal,
                                          "CNN_variant1_normal",
                                          lr=LR[3],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[3])
    print(f"Tiempo de entrenamiento CNN Variante 1 (Modelo 4) sin data augmentation: "
          f"{history_cnn_var1_normal['training_time']:.2f} segundos")


    # Entrenamos la Variante 1 (Modelo 4) con data augmentation
    print("\n--- Entrenando CNN Variante 1 (Modelo 4) con data augmentation ---")
    history_cnn_var1_aug = train_model(CNNVariant1, train_loader_aug,
                                          "CNN_variant1_aug",
                                          lr=LR[3],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[3])
    print(f"Tiempo de entrenamiento CNN Variante 1 (Modelo 4) con data augmentation: "
          f"{history_cnn_var1_aug['training_time']:.2f} segundos")

    # Entrenamos la Variante 2 (Modelo 5) sin data augmentation
    print("\n--- Entrenando CNN Variante 2 (Modelo 5) sin data augmentation ---")
    history_cnn_var2_normal = train_model(CNNVariant2, train_loader_normal,
                                          "CNN_variant2_normal",
                                          lr=LR[4],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[4])
    print(f"Tiempo de entrenamiento CNN Variante 2 (Modelo 5) sin data augmentation: "
          f"{history_cnn_var2_normal['training_time']:.2f} segundos")


    # Entrenamos la Variante 2 (Modelo 5) con data augmentation
    print("\n--- Entrenando CNN Variante 2 (Modelo 5) con data augmentation ---")
    history_cnn_var2_aug = train_model(CNNVariant2, train_loader_aug,
                                          "CNN_variant2_aug",
                                          lr=LR[4],
                                          wd=WEIGHT_DECAY,
                                          patience=PATIENCE[4])
    print(f"Tiempo de entrenamiento CNN Variante 2 (Modelo 5) con data augmentation: "
          f"{history_cnn_var2_aug['training_time']:.2f} segundos")

    print(f"Fecha y hora de finalización del entrenamiento de los modelos CNN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Guardando...\n")
    # Guardamos los historiales de los modelos CNN
    with open(os.path.join(RESULTS_DIR, 'historiales_modelos_cnn.pkl'), 'wb') as f:
        pkl.dump({
            'CNN_base_normal': history_cnn_base_normal,
            'CNN_base_aug': history_cnn_base_aug,
            'CNN_variant1_normal': history_cnn_var1_normal,
            'CNN_variant1_aug': history_cnn_var1_aug,
            'CNN_variant2_normal': history_cnn_var2_normal,
            'CNN_variant2_aug': history_cnn_var2_aug
        }, f) # type: ignore

    print("Entrenamiento de modelos CNN completado.")
    print(f"Resultados guardados en {RESULTS_DIR}/")

    # Resumen de los mejores resultados
    print("\n=== Resumen de resultados ===")
    print("Sin data augmentation:")
    print(f"CNN Base (normal): Mejor exactitud de validación = {max(history_cnn_base_normal['val_acc']):.4f}")
    print(f"CNN Variante 1 (normal): Mejor exactitud de validación = {max(history_cnn_var1_normal['val_acc']):.4f}")
    print(f"CNN Variante 2 (normal): Mejor exactitud de validación = {max(history_cnn_var2_normal['val_acc']):.4f}")
    print("\nCon data augmentation:")
    print(f"CNN Base (aug): Mejor exactitud de validación = {max(history_cnn_base_aug['val_acc']):.4f}")
    print(f"CNN Variante 1 (aug): Mejor exactitud de validación = {max(history_cnn_var1_aug['val_acc']):.4f}")
    print(f"CNN Variante 2 (aug): Mejor exactitud de validación = {max(history_cnn_var2_aug['val_acc']):.4f}")

    # Calculamos la mejora proporcionada por data augmentation
    print("\nMejora debido al data augmentation:")
    print(f"CNN Base: {(max(history_cnn_base_aug['val_acc']) - max(history_cnn_base_normal['val_acc'])) * 100:.2f}%")
    print(
        f"CNN Variante 1: {(max(history_cnn_var1_aug['val_acc']) - max(history_cnn_var1_normal['val_acc'])) * 100:.2f}%")
    print(
        f"CNN Variante 2: {(max(history_cnn_var2_aug['val_acc']) - max(history_cnn_var2_normal['val_acc'])) * 100:.2f}%")

    # Análisis de tiempo de ejecución
    print(f"Fecha y hora de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Guardamos los resultados de test en un diccionario
    test_results = {}

    # Después de entrenar NET-1 (sin augmentation)
    print("\n--- Evaluando NET-1 (sin augmentation) en test ---")
    model_net1_normal = NET1(input_size)
    model_net1_normal.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET1_normal_best.pth"), weights_only=True))
    test_metrics_net1_normal = evaluate_model(model_net1_normal, test_loader, nn.BCELoss(), "NET1_normal")
    test_results['NET1_normal'] = test_metrics_net1_normal

    # Mover model a CPU y liberar memoria
    model_net1_normal.to('cpu')
    del model_net1_normal
    liberar_memoria()

    # Después de entrenar NET-1 (con augmentation)
    print("\n--- Evaluando NET-1 (con augmentation) en test ---")
    model_net1_aug = NET1(input_size)
    model_net1_aug.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET1_aug_best.pth"), weights_only=True))
    test_metrics_net1_aug = evaluate_model(model_net1_aug, test_loader, nn.BCELoss(), "NET1_aug")
    test_results['NET1_aug'] = test_metrics_net1_aug

    # Mover model a CPU y liberar memoria
    model_net1_aug.to('cpu')
    del model_net1_aug
    liberar_memoria()

    # Después de entrenar NET-2 (sin augmentation)
    print("\n--- Evaluando NET-2 (sin augmentation) en test ---")
    model_net2_normal = NET2(input_size)
    model_net2_normal.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET2_normal_best.pth"), weights_only=True))
    test_metrics_net2_normal = evaluate_model(model_net2_normal, test_loader, nn.BCELoss(), "NET2_normal")
    test_results['NET2_normal'] = test_metrics_net2_normal

    # Mover model a CPU y liberar memoria
    model_net2_normal.to('cpu')
    del model_net2_normal
    liberar_memoria()

    # Después de entrenar NET-2 (con augmentation)
    print("\n--- Evaluando NET-2 (con augmentation) en test ---")
    model_net2_aug = NET2(input_size)
    model_net2_aug.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET2_aug_best.pth"), weights_only=True))
    test_metrics_net2_aug = evaluate_model(model_net2_aug, test_loader, nn.BCELoss(), "NET2_aug")
    test_results['NET2_aug'] = test_metrics_net2_aug

    # Mover model a CPU y liberar memoria
    model_net2_aug.to('cpu')
    del model_net2_aug
    liberar_memoria()

    # Guardamos los resultados de los modelos básicos
    with open(os.path.join(RESULTS_DIR, 'test_metrics_basicos.pkl'), 'wb') as f:
        pkl.dump({
            'NET1_normal': test_metrics_net1_normal,
            'NET1_aug': test_metrics_net1_aug,
            'NET2_normal': test_metrics_net2_normal,
            'NET2_aug': test_metrics_net2_aug
        }, f) # type: ignore

    # Después de entrenar CNN Base (sin augmentation)
    print("\n--- Evaluando CNN Base (sin augmentation) en test ---")
    model_cnn_base_normal = CNNBase()
    model_cnn_base_normal.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "CNN_base_normal_best.pth"), weights_only=True))
    test_metrics_cnn_base_normal = evaluate_model(model_cnn_base_normal, test_loader, nn.BCELoss(), "CNN_base_normal")
    test_results['CNN_base_normal'] = test_metrics_cnn_base_normal

    # Mover model a CPU y liberar memoria
    model_cnn_base_normal.to('cpu')
    del model_cnn_base_normal
    liberar_memoria()

    # Después de entrenar CNN Base (con augmentation)
    print("\n--- Evaluando CNN Base (con augmentation) en test ---")
    model_cnn_base_aug = CNNBase()
    model_cnn_base_aug.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "CNN_base_aug_best.pth"), weights_only=True))
    test_metrics_cnn_base_aug = evaluate_model(model_cnn_base_aug, test_loader, nn.BCELoss(), "CNN_base_aug")
    test_results['CNN_base_aug'] = test_metrics_cnn_base_aug

    # Mover model a CPU y liberar memoria
    model_cnn_base_aug.to('cpu')
    del model_cnn_base_aug
    liberar_memoria()

    # Después de entrenar CNN Variante 1 (sin augmentation)
    print("\n--- Evaluando CNN Variante 1 (sin augmentation) en test ---")
    model_cnn_var1_normal = CNNVariant1()
    model_cnn_var1_normal.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "CNN_variant1_normal_best.pth"), weights_only=True))
    test_metrics_cnn_var1_normal = evaluate_model(model_cnn_var1_normal, test_loader, nn.BCELoss(),
                                                  "CNN_variant1_normal")
    test_results['CNN_variant1_normal'] = test_metrics_cnn_var1_normal

    # Mover model a CPU y liberar memoria
    model_cnn_var1_normal.to('cpu')
    del model_cnn_var1_normal
    liberar_memoria()

    # Después de entrenar CNN Variante 1 (con augmentation)
    print("\n--- Evaluando CNN Variante 1 (con augmentation) en test ---")
    model_cnn_var1_aug = CNNVariant1()
    model_cnn_var1_aug.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "CNN_variant1_aug_best.pth"), weights_only=True))
    test_metrics_cnn_var1_aug = evaluate_model(model_cnn_var1_aug, test_loader, nn.BCELoss(), "CNN_variant1_aug")
    test_results['CNN_variant1_aug'] = test_metrics_cnn_var1_aug

    # Mover model a CPU y liberar memoria
    model_cnn_var1_aug.to('cpu')
    del model_cnn_var1_aug
    liberar_memoria()

    # Después de entrenar CNN Variante 2 (sin augmentation)
    print("\n--- Evaluando CNN Variante 2 (sin augmentation) en test ---")
    model_cnn_var2_normal = CNNVariant2()
    model_cnn_var2_normal.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant2_normal_best.pth"), weights_only=True))
    test_metrics_cnn_var2_normal = evaluate_model(model_cnn_var2_normal, test_loader, nn.BCELoss(),
                                                  "CNN_variant2_normal")
    test_results['CNN_variant2_normal'] = test_metrics_cnn_var2_normal

    # Mover model a CPU y liberar memoria
    model_cnn_var2_normal.to('cpu')
    del model_cnn_var2_normal
    liberar_memoria()

    # Después de entrenar CNN Variante 2 (con augmentation)
    print("\n--- Evaluando CNN Variante 2 (con augmentation) en test ---")
    model_cnn_var2_aug = CNNVariant2()
    model_cnn_var2_aug.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant2_aug_best.pth"), weights_only=True))
    test_metrics_cnn_var2_aug = evaluate_model(model_cnn_var2_aug, test_loader, nn.BCELoss(), "CNN_variant2_aug")
    test_results['CNN_variant2_aug'] = test_metrics_cnn_var2_aug

    # Mover model a CPU y liberar memoria
    model_cnn_var2_aug.to('cpu')
    del model_cnn_var2_aug
    liberar_memoria()

    # Guardamos todos los resultados de test de modelos optimizados (best)
    with open(os.path.join(RESULTS_DIR, 'test_metrics_cnn.pkl'), 'wb') as f:
        pkl.dump({
            'CNN_base_normal': test_metrics_cnn_base_normal,
            'CNN_base_aug': test_metrics_cnn_base_aug,
            'CNN_variant1_normal': test_metrics_cnn_var1_normal,
            'CNN_variant1_aug': test_metrics_cnn_var1_aug,
            'CNN_variant2_normal': test_metrics_cnn_var2_normal,
            'CNN_variant2_aug': test_metrics_cnn_var2_aug
        }, f)  # type: ignore

    # =====================================================================
    # EVALUACIÓN DE MODELOS FINALES (ENTRENADOS SIN EARLY STOPPING)
    # =====================================================================
    print("\n=== Evaluación de modelos finales (completados sin early stopping) ===")

    # Guardamos los resultados de test en un diccionario para modelos finales
    test_results_final = {}

    # Después de entrenar NET-1 (sin augmentation) - FINAL
    print("\n--- Evaluando NET-1 FINAL (sin augmentation) en test ---")
    model_net1_normal_final = NET1(input_size)
    model_net1_normal_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "NET1_normal_final.pth"), weights_only=True))
    test_metrics_net1_normal_final = evaluate_model(model_net1_normal_final, test_loader, nn.BCELoss(),
                                                    "NET1_normal_final")
    test_results_final['NET1_normal'] = test_metrics_net1_normal_final

    # Mover model a CPU y liberar memoria
    model_net1_normal_final.to('cpu')
    del model_net1_normal_final
    liberar_memoria()

    # Después de entrenar NET-1 (con augmentation) - FINAL
    print("\n--- Evaluando NET-1 FINAL (con augmentation) en test ---")
    model_net1_aug_final = NET1(input_size)
    model_net1_aug_final.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET1_aug_final.pth"), weights_only=True))
    test_metrics_net1_aug_final = evaluate_model(model_net1_aug_final, test_loader, nn.BCELoss(), "NET1_aug_final")
    test_results_final['NET1_aug'] = test_metrics_net1_aug_final

    # Mover model a CPU y liberar memoria
    model_net1_aug_final.to('cpu')
    del model_net1_aug_final
    liberar_memoria()

    # Después de entrenar NET-2 (sin augmentation) - FINAL
    print("\n--- Evaluando NET-2 FINAL (sin augmentation) en test ---")
    model_net2_normal_final = NET2(input_size)
    model_net2_normal_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "NET2_normal_final.pth"), weights_only=True))
    test_metrics_net2_normal_final = evaluate_model(model_net2_normal_final, test_loader, nn.BCELoss(),
                                                    "NET2_normal_final")
    test_results_final['NET2_normal'] = test_metrics_net2_normal_final

    # Mover model a CPU y liberar memoria
    model_net2_normal_final.to('cpu')
    del model_net2_normal_final
    liberar_memoria()

    # Después de entrenar NET-2 (con augmentation) - FINAL
    print("\n--- Evaluando NET-2 FINAL (con augmentation) en test ---")
    model_net2_aug_final = NET2(input_size)
    model_net2_aug_final.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "NET2_aug_final.pth"), weights_only=True))
    test_metrics_net2_aug_final = evaluate_model(model_net2_aug_final, test_loader, nn.BCELoss(), "NET2_aug_final")
    test_results_final['NET2_aug'] = test_metrics_net2_aug_final

    # Mover model a CPU y liberar memoria
    model_net2_aug_final.to('cpu')
    del model_net2_aug_final
    liberar_memoria()

    # Guardamos los resultados de los modelos básicos finales
    with open(os.path.join(RESULTS_DIR, 'test_metrics_basicos_final.pkl'), 'wb') as f:
        pkl.dump({
            'NET1_normal': test_metrics_net1_normal_final,
            'NET1_aug': test_metrics_net1_aug_final,
            'NET2_normal': test_metrics_net2_normal_final,
            'NET2_aug': test_metrics_net2_aug_final
        }, f)  # type: ignore

    # Después de entrenar CNN Base (sin augmentation) - FINAL
    print("\n--- Evaluando CNN Base FINAL (sin augmentation) en test ---")
    model_cnn_base_normal_final = CNNBase()
    model_cnn_base_normal_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_base_normal_final.pth"), weights_only=True))
    test_metrics_cnn_base_normal_final = evaluate_model(model_cnn_base_normal_final, test_loader, nn.BCELoss(),
                                                        "CNN_base_normal_final")
    test_results_final['CNN_base_normal'] = test_metrics_cnn_base_normal_final

    # Mover model a CPU y liberar memoria
    model_cnn_base_normal_final.to('cpu')
    del model_cnn_base_normal_final
    liberar_memoria()

    # Después de entrenar CNN Base (con augmentation) - FINAL
    print("\n--- Evaluando CNN Base FINAL (con augmentation) en test ---")
    model_cnn_base_aug_final = CNNBase()
    model_cnn_base_aug_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_base_aug_final.pth"), weights_only=True))
    test_metrics_cnn_base_aug_final = evaluate_model(model_cnn_base_aug_final, test_loader, nn.BCELoss(),
                                                     "CNN_base_aug_final")
    test_results_final['CNN_base_aug'] = test_metrics_cnn_base_aug_final

    # Mover model a CPU y liberar memoria
    model_cnn_base_aug_final.to('cpu')
    del model_cnn_base_aug_final
    liberar_memoria()

    # Después de entrenar CNN Variante 1 (sin augmentation) - FINAL
    print("\n--- Evaluando CNN Variante 1 FINAL (sin augmentation) en test ---")
    model_cnn_var1_normal_final = CNNVariant1()
    model_cnn_var1_normal_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant1_normal_final.pth"), weights_only=True))
    test_metrics_cnn_var1_normal_final = evaluate_model(model_cnn_var1_normal_final, test_loader, nn.BCELoss(),
                                                        "CNN_variant1_normal_final")
    test_results_final['CNN_variant1_normal'] = test_metrics_cnn_var1_normal_final

    # Mover model a CPU y liberar memoria
    model_cnn_var1_normal_final.to('cpu')
    del model_cnn_var1_normal_final
    liberar_memoria()

    # Después de entrenar CNN Variante 1 (con augmentation) - FINAL
    print("\n--- Evaluando CNN Variante 1 FINAL (con augmentation) en test ---")
    model_cnn_var1_aug_final = CNNVariant1()
    model_cnn_var1_aug_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant1_aug_final.pth"), weights_only=True))
    test_metrics_cnn_var1_aug_final = evaluate_model(model_cnn_var1_aug_final, test_loader, nn.BCELoss(),
                                                     "CNN_variant1_aug_final")
    test_results_final['CNN_variant1_aug'] = test_metrics_cnn_var1_aug_final

    # Mover model a CPU y liberar memoria
    model_cnn_var1_aug_final.to('cpu')
    del model_cnn_var1_aug_final
    liberar_memoria()

    # Después de entrenar CNN Variante 2 (sin augmentation) - FINAL
    print("\n--- Evaluando CNN Variante 2 FINAL (sin augmentation) en test ---")
    model_cnn_var2_normal_final = CNNVariant2()
    model_cnn_var2_normal_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant2_normal_final.pth"), weights_only=True))
    test_metrics_cnn_var2_normal_final = evaluate_model(model_cnn_var2_normal_final, test_loader, nn.BCELoss(),
                                                        "CNN_variant2_normal_final")
    test_results_final['CNN_variant2_normal'] = test_metrics_cnn_var2_normal_final

    # Mover model a CPU y liberar memoria
    model_cnn_var2_normal_final.to('cpu')
    del model_cnn_var2_normal_final
    liberar_memoria()

    # Después de entrenar CNN Variante 2 (con augmentation) - FINAL
    print("\n--- Evaluando CNN Variante 2 FINAL (con augmentation) en test ---")
    model_cnn_var2_aug_final = CNNVariant2()
    model_cnn_var2_aug_final.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "CNN_variant2_aug_final.pth"), weights_only=True))
    test_metrics_cnn_var2_aug_final = evaluate_model(model_cnn_var2_aug_final, test_loader, nn.BCELoss(),
                                                     "CNN_variant2_aug_final")
    test_results_final['CNN_variant2_aug'] = test_metrics_cnn_var2_aug_final

    # Mover model a CPU y liberar memoria
    model_cnn_var2_aug_final.to('cpu')
    del model_cnn_var2_aug_final
    liberar_memoria()

    # Guardamos todos los resultados de test de modelos finales de CNN
    with open(os.path.join(RESULTS_DIR, 'test_metrics_cnn_final.pkl'), 'wb') as f:
        pkl.dump({
            'CNN_base_normal': test_metrics_cnn_base_normal_final,
            'CNN_base_aug': test_metrics_cnn_base_aug_final,
            'CNN_variant1_normal': test_metrics_cnn_var1_normal_final,
            'CNN_variant1_aug': test_metrics_cnn_var1_aug_final,
            'CNN_variant2_normal': test_metrics_cnn_var2_normal_final,
            'CNN_variant2_aug': test_metrics_cnn_var2_aug_final
        }, f)  # type: ignore

    # Resumen comparativo de resultados best vs final
    print("\n=== Comparativa modelos optimizados (best) vs modelos completos (final) ===")
    for modelo in test_results.keys():
        best_acc = test_results[modelo]['test_acc']
        final_acc = test_results_final[modelo]['test_acc']
        diff = (final_acc - best_acc) * 100
        print(f"{modelo}: Best {best_acc:.4f} vs Final {final_acc:.4f} (Diff: {diff:.2f}%)")

    # ================================================================================
    # RESUMEN DE RESULTADOS DE LOS MODELOS OPTIMIZADOS (BEST) y DE LOS MODELOS FINALES
    # ================================================================================

    print("\n=== Comparativa modelos optimizados (best) vs modelos completos (final) ===")
    for modelo in test_results.keys():
        best_acc = test_results[modelo]['test_acc']
        final_acc = test_results_final[modelo]['test_acc']
        diff = (final_acc - best_acc) * 100
        print(f"{modelo}: Best {best_acc:.4f} vs Final {final_acc:.4f} (Diff: {diff:.2f}%)")

    # Usar la función para imprimir resúmenes
    print_test_summary(test_results, "Resumen de resultados en test")
    print_test_summary(test_results_final, "Resumen de resultados en test", is_final=True)