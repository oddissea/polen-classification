# ************************************************************************
# * analysis/data_augmentation.py
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
Módulo para el análisis del impacto de data augmentation en los modelos.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from .visualization import configure_output_dir
from .performance import analizar_matrices_confusion, analizar_metricas_por_clase


def analizar_errores_por_clase(modelos, dataloaders, test_metrics, nombres_modelos, results_dir='resultados'):
    """
    Analiza ejemplos de errores específicos para entender mejor el impacto de data augmentation.

    Args:
        modelos: Diccionario con los modelos cargados
        dataloaders: Diccionario con los dataloaders
        test_metrics: Diccionario con métricas de test
        nombres_modelos: Lista de nombres de modelos a analizar
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'errores')
    print("\n--- Análisis de errores específicos ---")

    # Verificamos si tenemos modelos y datos de test
    if not modelos or not dataloaders.get('test'):
        print("No hay modelos o datos de test disponibles para analizar errores.")
        return

    test_data = dataloaders['test']
    device = next(iter(modelos.values())).parameters().__next__().device

    # Para cada modelo, analizamos ejemplos mal clasificados
    for nombre_modelo in nombres_modelos:
        if nombre_modelo not in modelos or nombre_modelo not in test_metrics:
            continue

        model = modelos[nombre_modelo]
        model.eval()

        # Recolectamos ejemplos mal clasificados
        falsas_kunzea = []  # Ejemplos de Lepto clasificados como Kunzea
        falsos_lepto = []  # Ejemplos de Kunzea clasificados como Lepto

        with torch.no_grad():
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predicted = (outputs > 0.5).float().squeeze()

                # Buscamos ejemplos mal clasificados
                for index in range(len(labels)):
                    if labels[index] == 1 and predicted[index] == 0:  # Lepto clasificado como Kunzea
                        falsas_kunzea.append((images[index].cpu(), labels[index].item()))
                    elif labels[index] == 0 and predicted[index] == 1:  # Kunzea clasificado como Lepto
                        falsos_lepto.append((images[index].cpu(), labels[index].item()))

                # Limitamos a unos pocos ejemplos para no hacer la visualización demasiado grande
                if len(falsas_kunzea) >= 5 and len(falsos_lepto) >= 5:
                    break

        # Visualizamos ejemplos mal clasificados
        num_ejemplos = min(5, min(len(falsas_kunzea), len(falsos_lepto)))
        if num_ejemplos > 0:
            plt.figure(figsize=(10, 4 * num_ejemplos))

            for index in range(num_ejemplos):
                # Falsas Kunzea
                plt.subplot(num_ejemplos, 2, 2 * index + 1)
                plt.imshow(falsas_kunzea[index][0].squeeze(), cmap='gray')
                plt.title(f"Lepto clasificado como Kunzea")
                plt.axis('off')

                # Falsos Lepto
                plt.subplot(num_ejemplos, 2, 2 * index + 2)
                plt.imshow(falsos_lepto[index][0].squeeze(), cmap='gray')
                plt.title(f"Kunzea clasificado como Lepto")
                plt.axis('off')

            plt.suptitle(f"Ejemplos mal clasificados - {nombre_modelo}")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_results_dir, f'errores_{nombre_modelo}.png'), dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\nModelo: {nombre_modelo}")
            print(f"Total de ejemplos Lepto clasificados como Kunzea: {len(falsas_kunzea)}")
            print(f"Total de ejemplos Kunzea clasificados como Lepto: {len(falsos_lepto)}")

            # Calculamos la proporción de errores
            conf_matrix = test_metrics[nombre_modelo]['conf_matrix']
            tn, fp, fn, tp = conf_matrix.ravel()

            error_kunzea = fp / (tn + fp) if (tn + fp) > 0 else 0
            error_lepto = fn / (tp + fn) if (tp + fn) > 0 else 0

            print(f"Tasa de error para Kunzea: {error_kunzea:.4f} ({fp}/{tn + fp})")
            print(f"Tasa de error para Lepto: {error_lepto:.4f} ({fn}/{tp + fn})")
            print(f"Desequilibrio entre tasas de error: {abs(error_kunzea - error_lepto):.4f}")

            # Guardamos las métricas de error en CSV
            error_stats = {
                'Modelo': [nombre_modelo],
                'Total_Falsas_Kunzea': [len(falsas_kunzea)],
                'Total_Falsos_Lepto': [len(falsos_lepto)],
                'Tasa_Error_Kunzea': [error_kunzea],
                'Tasa_Error_Lepto': [error_lepto],
                'Desequilibrio_Error': [abs(error_kunzea - error_lepto)]
            }

            error_df = pd.DataFrame(error_stats)
            error_csv_path = os.path.join(vis_results_dir, f'statistics_errores_{nombre_modelo}.csv')
            error_df.to_csv(error_csv_path, index=False)
            print(f"Estadísticas de errores guardadas en: {error_csv_path}")

        else:
            print(f"\nModelo: {nombre_modelo} - No se encontraron suficientes ejemplos mal clasificados para analizar.")


def _calcular_metricas_augmentation(test_metrics, base):
    """
    Función auxiliar que calcula las métricas comparativas de un modelo con y sin data augmentation.

    Args:
        test_metrics: Diccionario con todas las métricas de test
        base: Nombre base del modelo (sin sufijos '_normal'/'_aug')

    Returns:
        dict: Diccionario con las métricas comparativas calculadas o None si faltan datos
    """
    normal_key = f"{base}_normal"
    aug_key = f"{base}_aug"

    if normal_key not in test_metrics or aug_key not in test_metrics:
        return None

    # Obtenemos las métricas
    metrics_normal = test_metrics[normal_key]
    metrics_aug = test_metrics[aug_key]

    # Calculamos la mejora absoluta
    mejora_acc = metrics_aug['test_acc'] - metrics_normal['test_acc']
    mejora_f1 = metrics_aug['f1'] - metrics_normal['f1']

    # Analizamos la matriz de confusión
    conf_normal = metrics_normal['conf_matrix']
    conf_aug = metrics_aug['conf_matrix']

    tn_normal, fp_normal, fn_normal, tp_normal = conf_normal.ravel()
    tn_aug, fp_aug, fn_aug, tp_aug = conf_aug.ravel()

    # Calculamos exactitud por clase
    acc_kunzea_normal = tn_normal / (tn_normal + fp_normal) if (tn_normal + fp_normal) > 0 else 0
    acc_kunzea_aug = tn_aug / (tn_aug + fp_aug) if (tn_aug + fp_aug) > 0 else 0
    acc_lepto_normal = tp_normal / (tp_normal + fn_normal) if (tp_normal + fn_normal) > 0 else 0
    acc_lepto_aug = tp_aug / (tp_aug + fn_aug) if (tp_aug + fn_aug) > 0 else 0

    # Calculamos el desequilibrio
    deseq_normal = abs(acc_kunzea_normal - acc_lepto_normal)
    deseq_aug = abs(acc_kunzea_aug - acc_lepto_aug)

    # Retornamos un diccionario con todas las métricas
    return {
        'Modelo': base,
        'Exactitud_Normal': metrics_normal['test_acc'],
        'Exactitud_Aug': metrics_aug['test_acc'],
        'Mejora_Exactitud': mejora_acc,
        'F1_Normal': metrics_normal['f1'],
        'F1_Aug': metrics_aug['f1'],
        'Mejora_F1': mejora_f1,
        'Exactitud_Kunzea_Normal': acc_kunzea_normal,
        'Exactitud_Kunzea_Aug': acc_kunzea_aug,
        'Mejora_Kunzea': acc_kunzea_aug - acc_kunzea_normal,
        'Exactitud_Lepto_Normal': acc_lepto_normal,
        'Exactitud_Lepto_Aug': acc_lepto_aug,
        'Mejora_Lepto': acc_lepto_aug - acc_lepto_normal,
        'Desequilibrio_Normal': deseq_normal,
        'Desequilibrio_Aug': deseq_aug,
        'Mejora_Equilibrio': deseq_normal - deseq_aug,
        'Equilibrio_Mejorado': deseq_aug < deseq_normal
    }

def analisis_cualitativo_completo(test_metrics, modelos, dataloaders=None, results_dir='resultados'):
    """
    Realiza un análisis cualitativo completo comparando los modelos con y sin data augmentation.

    Args:
        test_metrics: Diccionario con todas las métricas de test
        modelos: Diccionario con los modelos cargados
        dataloaders: Diccionario con los dataloaders (opcional)
        results_dir: Directorio donde guardar los resultados
    """

    # Configuramos el directorio de visualizaciones para el resumen
    vis_results_dir = configure_output_dir(results_dir, 'resumen')

    # Definimos las bases de los modelos
    modelos_base = ['NET1', 'NET2', 'CNN_base', 'CNN_variant1', 'CNN_variant2']

    # 1. Análisis de matrices de confusión
    analizar_matrices_confusion(test_metrics, modelos_base, results_dir)

    # 2. Análisis de métricas por clase
    analizar_metricas_por_clase(test_metrics, modelos_base, results_dir)

    # 3. Análisis de errores específicos
    # Solo analizamos algunos modelos clave para no saturar la visualización
    modelos_para_analisis_errores = ['CNN_base_normal', 'CNN_base_aug', 'CNN_variant1_normal', 'CNN_variant1_aug']

    # Verificamos si tenemos acceso a los dataloaders
    if dataloaders and 'test' in dataloaders:
        analizar_errores_por_clase(modelos, dataloaders, test_metrics, modelos_para_analisis_errores, results_dir)
    else:
        print("No hay dataloader de test disponible. El análisis de errores no se realizará.")

    # 4. Conclusiones comparativas del data augmentation
    print("\n--- Resumen del impacto cualitativo del data augmentation ---")

    resultados_data = []

    for base in modelos_base:
        # Calculamos las métricas para este modelo
        metricas = _calcular_metricas_augmentation(test_metrics, base)

        if metricas:
            resultados_data.append(metricas)

            # Imprimimos el resumen
            print(f"\nModelo: {base}")
            print(
                f"Mejora absoluta en exactitud: {metricas['Mejora_Exactitud']:.4f} ({metricas['Mejora_Exactitud'] * 100:.2f}%)")
            print(f"Mejora absoluta en F1-Score: {metricas['Mejora_F1']:.4f} ({metricas['Mejora_F1'] * 100:.2f}%)")
            print(f"Mejora en Kunzea: {metricas['Mejora_Kunzea']:.4f} ({metricas['Mejora_Kunzea'] * 100:.2f}%)")
            print(f"Mejora en Lepto: {metricas['Mejora_Lepto']:.4f} ({metricas['Mejora_Lepto'] * 100:.2f}%)")

            # Conclusión sobre el equilibrio
            if metricas['Equilibrio_Mejorado']:
                print(
                    f"El data augmentation ha MEJORADO el equilibrio entre clases: {metricas['Desequilibrio_Normal']:.4f} → {metricas['Desequilibrio_Aug']:.4f}")
                print(
                    f"Reducción del desequilibrio: {(1 - metricas['Desequilibrio_Aug'] / metricas['Desequilibrio_Normal']) * 100:.2f}%")
            else:
                print(
                    f"El data augmentation ha EMPEORADO el equilibrio entre clases: {metricas['Desequilibrio_Normal']:.4f} → {metricas['Desequilibrio_Aug']:.4f}")
                print(
                    f"Aumento del desequilibrio: {(metricas['Desequilibrio_Aug'] / metricas['Desequilibrio_Normal'] - 1) * 100:.2f}%")

            # Conclusión general
            if metricas['Mejora_Exactitud'] > 0:
                if metricas['Equilibrio_Mejorado']:
                    print(
                        "CONCLUSIÓN: El data augmentation mejoró tanto la exactitud global como el equilibrio entre clases.")
                else:
                    print(
                        "CONCLUSIÓN: El data augmentation mejoró la exactitud global pero a costa de un mayor desequilibrio entre clases.")
            else:
                if metricas['Equilibrio_Mejorado']:
                    print(
                        "CONCLUSIÓN: El data augmentation empeoró la exactitud global pero mejoró el equilibrio entre clases.")
                else:
                    print(
                        "CONCLUSIÓN: El data augmentation empeoró tanto la exactitud global como el equilibrio entre clases.")

    # Creamos y guardamos el DataFrame
    if resultados_data:
        resumen_df = pd.DataFrame(resultados_data)
        resumen_csv_path = os.path.join(vis_results_dir, 'resumen_impacto_augmentation.csv')
        resumen_df.to_csv(resumen_csv_path, index=False)
        print(f"Resumen del impacto del data augmentation guardado en: {resumen_csv_path}")


def visualizar_transformaciones_augmentation(imagen_original, transformaciones, num_ejemplos=5,
                                             results_dir='resultados'):
    """
    Visualiza varias ejecuciones de la transformación de data augmentation.

    Args:
        imagen_original: Imagen original a transformar (PIL Image)
        transformaciones: Función de transformación de PyTorch
        num_ejemplos: Número de ejemplos transformados a mostrar
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'transformaciones')

    # Configuramos la semilla para reproducibilidad
    torch.manual_seed(42)

    # Mostramos la imagen original y varias ejecuciones de las transformaciones
    plt.figure(figsize=(15, 3))

    # Imagen original
    plt.subplot(1, num_ejemplos + 1, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Aplicamos la transformación completa varias veces con diferentes semillas
    for i in range(num_ejemplos):
        # Usamos una semilla diferente cada vez para obtener variaciones
        torch.manual_seed(42 + i)

        # Aplicamos las transformaciones completas (las mismas que se usan en entrenamiento)
        imagen_aug = transformaciones(imagen_original)

        # Convertimos a numpy para visualización
        if torch.is_tensor(imagen_aug):
            imagen_np = imagen_aug.numpy().squeeze()
        else:
            imagen_np = np.array(imagen_aug)

        # Mostramos la imagen transformada
        plt.subplot(1, num_ejemplos + 1, i + 2)
        plt.imshow(imagen_np, cmap='gray')
        plt.title(f"Variación {i + 1}")
        plt.axis('off')

    plt.suptitle("Variaciones aleatorias usando la transformación de entrenamiento", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, 'variaciones_augmentation.png'), dpi=300, bbox_inches='tight')
    plt.show()


def analizar_data_augmentation_detallado(imagen_original, transformaciones, results_dir='resultados'):
    """
    Realiza un análisis detallado de las diferentes transformaciones de data augmentation.

    Args:
        imagen_original: Imagen original a transformar (PIL Image)
        transformaciones: Diccionario con transformaciones individuales
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'transformaciones')

    # Configuramos la semilla para reproducibilidad
    torch.manual_seed(42)

    # Mostramos la imagen original y cada transformación por separado
    plt.figure(figsize=(15, 3))

    # Número de transformaciones (incluyendo la original)
    num_transformaciones = len(transformaciones) + 1

    # Imagen original
    plt.subplot(1, num_transformaciones, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Aplicamos cada transformación por separado
    for i, (nombre, transformacion) in enumerate(transformaciones.items()):
        # Aplicamos la transformación
        imagen_transformada = transformacion(imagen_original)

        # Convertimos a numpy para visualización
        if torch.is_tensor(imagen_transformada):
            imagen_np = imagen_transformada.numpy().squeeze()
        else:
            imagen_np = np.array(imagen_transformada)

        # Mostramos la imagen transformada
        plt.subplot(1, num_transformaciones, i + 2)
        plt.imshow(imagen_np, cmap='gray')
        plt.title(nombre)
        plt.axis('off')

    plt.suptitle("Efecto de cada transformación individual", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, 'transformaciones_individuales.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Documentamos qué transformaciones se analizaron
    trans_data = [{'Nombre': nombre} for nombre in transformaciones.keys()]
    trans_df = pd.DataFrame(trans_data)
    trans_csv_path = os.path.join(vis_results_dir, 'transformaciones_analizadas.csv')
    trans_df.to_csv(trans_csv_path, index=False)
    print(f"Lista de transformaciones analizadas guardada en: {trans_csv_path}")