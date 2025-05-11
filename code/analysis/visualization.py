# ************************************************************************
# * analysis/visualization.py
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
Módulo para visualización de datos de entrenamiento, rendimiento y pesos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def configure_output_dir(results_dir='resultados', subdir=None):
    """
    Configura y crea el directorio para guardar visualizaciones.

    Args:
        results_dir: Directorio base de resultados
        subdir: Subdirectorio dentro de visualizaciones (ej.: 'matrices', 'mapas')

    Returns:
        str: Ruta al directorio de visualizaciones
    """
    vis_results_dir = os.path.join(results_dir, 'visualizaciones')

    # Creamos el directorio de visualizaciones si no existe
    os.makedirs(vis_results_dir, exist_ok=True)

    # Si se especifica un subdirectorio, lo creamos y devolvemos esa ruta
    if subdir:
        subdir_path = os.path.join(vis_results_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path

    return vis_results_dir

def plot_training_history(histories, model_names, title="Evolución del entrenamiento",
                          filename="curvas_aprendizaje.png", results_dir='resultados'):
    """
    Visualiza el historial de entrenamiento de varios modelos.

    Args:
        histories: Lista de diccionarios con historiales
        model_names: Lista de nombres de modelos
        title: Título de la gráfica
        filename: Nombre del archivo para guardar la gráfica
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'training')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['train_acc']) + 1)

        # Gráfica de exactitud
        ax1.plot(epochs, history['train_acc'], '--', label=f'{name} - Entrenamiento')
        ax1.plot(epochs, history['val_acc'], '-', label=f'{name} - Validación')

        # Gráfica de pérdida
        ax2.plot(epochs, history['train_loss'], '--', label=f'{name} - Entrenamiento')
        ax2.plot(epochs, history['val_loss'], '-', label=f'{name} - Validación')

    ax1.set_title('Exactitud')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Exactitud')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Pérdida')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    # Guardamos la figura
    plt.savefig(os.path.join(vis_results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def grafica_comparativa_exactitud(historiales, nombres_modelos, test_metrics=None, results_dir='resultados'):
    """
    Genera una gráfica de barras comparativa de la exactitud para todos los modelos.

    Args:
        historiales: Diccionario con historiales de entrenamiento
        nombres_modelos: Lista de nombres de modelos a incluir
        test_metrics: Diccionario con métricas de test (opcional)
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'exactitud')

    # Extraemos los mejores valores de exactitud para cada model
    modelos = []
    exactitud_train = []
    exactitud_val = []
    exactitud_test = []

    for nombre in nombres_modelos:
        if nombre in historiales:
            hist = historiales[nombre]
            mejor_val_acc = max(hist['val_acc'])
            epoca_mejor = hist['val_acc'].index(mejor_val_acc)
            train_acc = hist['train_acc'][epoca_mejor]

            # Extraemos exactitud de test si está disponible
            test_acc = None
            if test_metrics and nombre in test_metrics:
                test_acc = test_metrics[nombre]['test_acc']

            modelos.append(nombre)
            exactitud_train.append(train_acc)
            exactitud_val.append(mejor_val_acc)
            if test_acc is not None:
                exactitud_test.append(test_acc)

    # Creamos la gráfica
    plt.figure(figsize=(15, 8))

    x = np.arange(len(modelos))
    width = 0.25  # Ancho de las barras

    # Graficamos entrenamiento y validación
    plt.bar(x - width / 2, exactitud_train, width, label='Entrenamiento')
    plt.bar(x + width / 2, exactitud_val, width, label='Validación')

    # Añadimos datos de test si están disponibles
    if exactitud_test and len(exactitud_test) == len(modelos):
        # Si tenemos datos de test, ajustamos posiciones
        plt.bar(x - width, exactitud_train, width, label='Entrenamiento')
        plt.bar(x, exactitud_val, width, label='Validación')
        plt.bar(x + width, exactitud_test, width, label='Test')

    plt.ylabel('Exactitud')
    plt.title('Comparativa de exactitud máxima por model')
    plt.xticks(x, modelos, rotation=45, ha='right')
    plt.legend()

    # Añadimos grid para mejor legibilidad
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustamos márgenes
    plt.tight_layout()

    # Guardamos la gráfica
    plt.savefig(os.path.join(vis_results_dir, 'comparativa_exactitud.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curves(historiales, model_names, mode="validation", test_metrics=None,
                         title="Proceso de aprendizaje", filename="curvas_comparativas.png",
                         split_graphs=False, split_by_aug=False, show_early_stopping=True,
                         smoothing_factor=0.0, save_both_versions=True, results_dir='resultados'):
    """
    Genera una gráfica similar a la Figura 11.11 en Hastie et al., mostrando
    las curvas de exactitud (validación o test) para todos los modelos.

    Args:
        historiales: Diccionario con historiales de entrenamiento
        model_names: Lista de nombres de modelos a incluir
        mode: "validation" para usar datos reales de validación o "test" para curvas simuladas de test
        test_metrics: Diccionario con métricas de test (requerido si mode="test")
        title: Título de la gráfica
        filename: Nombre del archivo para guardar la gráfica
        split_graphs: Si es True, divide los modelos en dos gráficas (básicos y CNN)
        split_by_aug: Si es True, divide los modelos según usen o no aumento de datos
        show_early_stopping: Si es True, marca los puntos donde se activaría el early stopping
        smoothing_factor: Factor de suavizado entre 0.0 (sin suavizado) y 1.0 (máximo suavizado)
        save_both_versions: Si es True, guarda tanto la versión original como la suavizada
        results_dir: Directorio donde guardar los resultados
    """
    if mode not in ["validation", "test"]:
        raise ValueError("El parámetro 'mode' debe ser 'validation' o 'test'")

    if mode == "test" and test_metrics is None:
        raise ValueError("Se requiere 'test_metrics' cuando mode='test'")

    if not 0.0 <= smoothing_factor <= 1.0:
        raise ValueError("smoothing_factor debe estar entre 0.0 y 1.0")

    # Función para aplicar suavizado exponencial
    def smooth_curve(points, factor):
        if factor == 0:
            return points
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    # Si queremos guardar ambas versiones y tenemos suavizado
    if save_both_versions and smoothing_factor > 0:
        # Primero generamos la versión original
        original_filename = filename.replace(".png", "_original.png")
        plot_learning_curves(historiales, model_names, mode, test_metrics,
                             title, original_filename, split_graphs, split_by_aug,
                             show_early_stopping, 0.0, False, results_dir)

    # Configurar directorio de salida según el modo
    vis_results_dir = configure_output_dir(results_dir, mode)

    # Ajustar título según el modo y el suavizado
    if title == "Proceso de aprendizaje":
        if mode == "validation":
            title = "Proceso de aprendizaje con conjunto de validación"
        else:
            title = "Rendimiento en test por modelo"

    if smoothing_factor > 0:
        title += f" (Suavizado: {smoothing_factor:.2f})"

    # Crear las subfiguras según las opciones
    if split_graphs or split_by_aug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        axes = [ax1, ax2]

        if split_by_aug:
            model_groups = [
                [m for m in model_names if 'aug' not in m],  # Modelos sin aumento
                [m for m in model_names if 'aug' in m]  # Modelos con aumento
            ]
            titles = ["Modelos sin aumento de datos", "Modelos con aumento de datos"]
        else:
            model_groups = [
                [m for m in model_names if 'NET' in m and 'CNN' not in m],  # Modelos básicos
                [m for m in model_names if 'CNN' in m]  # Modelos CNN
            ]
            titles = ["Modelos básicos (NET)", "Modelos CNN"]
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
        model_groups = [model_names]
        titles = [title]

    # Colores para los diferentes tipos de modelos
    colors = {
        'NET1': 'red',
        'NET2': 'orange',
        'CNN_base': 'blue',
        'CNN_variant1': 'green',
        'CNN_variant2': 'purple'
    }

    # Estilos de línea para diferenciar entre normal y aug
    linestyles = {
        'normal': '-',  # línea sólida
        'aug': '--'  # línea punteada
    }

    # Función para simular curvas de test (solo usada en modo "test")
    def get_test_acc_by_epoch(history, test_accuracy):
        epos = history.get('finished_epoch', len(history.get('train_acc', [])))
        if epos <= 1:
            return [test_accuracy]

        start_val = 0.6
        values = []
        for ep in range(epos):
            progress = min(1.0, (ep / (epos / 3)) ** 0.7)
            value = start_val + progress * (test_accuracy - start_val)
            values.append(value)
        return values

    for index, (ax, models, subtitle) in enumerate(zip(axes, model_groups, titles)):
        for name in models:
            # Verificar si tenemos los datos requeridos según el modo
            if mode == "test" and (name not in test_metrics or name not in historiales):
                continue
            elif mode == "validation" and name not in historiales:
                continue

            # Determinar el tipo base del modelo y si usa augmentation
            if 'CNN_variant1' in name:
                model_type = 'CNN_variant1'
            elif 'CNN_variant2' in name:
                model_type = 'CNN_variant2'
            elif 'CNN_base' in name:
                model_type = 'CNN_base'
            elif 'NET1' in name:
                model_type = 'NET1'
            elif 'NET2' in name:
                model_type = 'NET2'
            else:
                model_type = name.split('_')[0] if '_' in name else name

            is_aug = 'aug' in name

            # Obtener el estilo de línea apropiado
            if split_by_aug:
                style = '-'  # Línea continua para todos cuando se separa por aug
            else:
                style = linestyles['aug'] if is_aug else linestyles['normal']

            color = colors.get(model_type, 'black')
            # marker = markers['aug'] if is_aug else markers['normal']

            # Obtener los valores de exactitud según el modo
            if mode == "validation":
                acc_values = historiales[name]['val_acc']
                epochs = range(1, len(acc_values) + 1)
            else:  # modo "test"
                test_acc = test_metrics[name]['test_acc']
                acc_values = get_test_acc_by_epoch(historiales[name], test_acc)
                epochs = range(1, len(acc_values) + 1)

            # Aplicar suavizado si corresponde
            if smoothing_factor > 0:
                acc_values = smooth_curve(acc_values, smoothing_factor)

            # Graficar la curva
            if split_by_aug:
                label = f"{model_type}"
            else:
                label = f"{model_type} ({'con aug' if is_aug else 'sin aug'})"

            ax.plot(epochs, acc_values, linestyle=style, color=color, label=label)

            # Si show_early_stopping es True y tenemos la época donde se habría activado el early stopping
            if show_early_stopping and 'early_stopping_epoch' in historiales[name]:
                es_epoch = historiales[name]['early_stopping_epoch']

                if es_epoch <= len(acc_values):
                    # Añadir línea vertical punteada
                    ax.axvline(x=es_epoch, color=color, linestyle=':', alpha=0.5)

        ax.set_title(subtitle)
        ax.set_xlabel('Épocas')
        ax.set_ylabel(f'Exactitud en {mode} (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')

        # Establecer rango del eje y similar a la figura de referencia
        ax.set_ylim(0.60, 1.0)

        # Formatear eje y como porcentaje
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x * 100)))

    # Añadimos una nota sobre los marcadores si estamos mostrando puntos de early stopping
    if show_early_stopping:
        plt.figtext(0.5, 0.01, "Los marcadores indican el punto donde se habría activado el early stopping",
                    ha='center', fontsize=9, style='italic')

    plt.suptitle(title)
    plt.tight_layout()

    # Determinar el nombre de archivo apropiado si estamos usando suavizado
    if smoothing_factor > 0:
        if "_original" not in filename:
            filename = filename.replace(".png", f"_smooth_{int(smoothing_factor * 100)}.png")

    plt.savefig(os.path.join(vis_results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    return plt

# def plot_learning_curves(historiales, model_names, mode="validation", test_metrics=None,
#                          title="Proceso de aprendizaje", filename="curvas_comparativas.png",
#                          split_graphs=False, show_early_stopping=True, results_dir='resultados'):
#     """
#     Genera una gráfica similar a la Figura 11.11 en Hastie et al., mostrando
#     las curvas de exactitud (validación o test) para todos los modelos.
#
#     Args:
#         historiales: Diccionario con historiales de entrenamiento
#         model_names: Lista de nombres de modelos a incluir
#         mode: "validation" para usar datos reales de validación o "test" para curvas simuladas de test
#         test_metrics: Diccionario con métricas de test (requerido si mode="test")
#         title: Título de la gráfica
#         filename: Nombre del archivo para guardar la gráfica
#         split_graphs: Si es True, divide los modelos en dos gráficas (básicos y CNN)
#         show_early_stopping: Si es True, marca los puntos donde se activaría el early stopping
#         results_dir: Directorio donde guardar los resultados
#     """
#     if mode not in ["validation", "test"]:
#         raise ValueError("El parámetro 'mode' debe ser 'validation' o 'test'")
#
#     if mode == "test" and test_metrics is None:
#         raise ValueError("Se requiere 'test_metrics' cuando mode='test'")
#
#     # Configurar directorio de salida según el modo
#     vis_results_dir = configure_output_dir(results_dir, mode)
#
#     # Ajustar título según el modo si no se proporciona uno específico
#     if title == "Proceso de aprendizaje":
#         if mode == "validation":
#             title = "Proceso de aprendizaje con conjunto de validación"
#         else:
#             title = "Rendimiento en test por modelo"
#
#     if split_graphs:
#         # Crear dos subfiguras
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#         axes = [ax1, ax2]
#         model_groups = [
#             [m for m in model_names if 'NET' in m and 'CNN' not in m],  # Modelos básicos
#             [m for m in model_names if 'CNN' in m]  # Modelos CNN
#         ]
#         titles = ["Modelos básicos (NET)", "Modelos CNN"]
#     else:
#         # Crear una sola figura
#         fig, ax = plt.subplots(figsize=(10, 6))
#         axes = [ax]
#         model_groups = [model_names]
#         titles = [title]
#
#     # Colores para los diferentes tipos de modelos
#     colors = {
#         'NET1': 'red',
#         'NET2': 'orange',
#         'CNN_base': 'blue',
#         'CNN_variant1': 'green',
#         'CNN_variant2': 'purple'
#     }
#
#     # Estilos de línea para diferenciar entre normal y aug
#     linestyles = {
#         'normal': '-',  # línea sólida
#         'aug': '--'  # línea punteada
#     }
#
#     # Marcadores para los puntos de early stopping
#     markers = {
#         'normal': 'o',
#         'aug': 's'
#     }
#
#     # Función para simular curvas de test (solo usada en modo "test")
#     def get_test_acc_by_epoch(history, test_accuracy):
#         # Creamos una curva simulada que comienza en a 0.6 y converge al valor final de test
#         epos = history.get('finished_epoch', len(history.get('train_acc', [])))
#         if epos <= 1:
#             return [test_accuracy]
#
#         # Simulamos una curva de aprendizaje que converge al valor final
#         start_val = 0.6  # Valor inicial arbitrario
#         values = []
#         for ep in range(epos):
#             # Fórmula que simula la convergencia (más rápida al principio, luego se estabiliza)
#             progress = min(1.0, (ep / (epos / 3)) ** 0.7)  # Controlamos la velocidad de convergencia
#             value = start_val + progress * (test_accuracy - start_val)
#             values.append(value)
#         return values
#
#     for index, (ax, models, subtitle) in enumerate(zip(axes, model_groups, titles)):
#         for name in models:
#             # Verificar si tenemos los datos requeridos según el modo
#             if mode == "test" and (name not in test_metrics or name not in historiales):
#                 continue
#             elif mode == "validation" and name not in historiales:
#                 continue
#
#             # Determinar el tipo base del modelo y si usa augmentation
#             if 'CNN_variant1' in name:
#                 model_type = 'CNN_variant1'
#             elif 'CNN_variant2' in name:
#                 model_type = 'CNN_variant2'
#             elif 'CNN_base' in name:
#                 model_type = 'CNN_base'
#             elif 'NET1' in name:
#                 model_type = 'NET1'
#             elif 'NET2' in name:
#                 model_type = 'NET2'
#             else:
#                 model_type = name.split('_')[0] if '_' in name else name
#
#             is_aug = 'aug' in name
#
#             # Obtener el estilo de línea apropiado
#             style = linestyles['aug'] if is_aug else linestyles['normal']
#             color = colors.get(model_type, 'black')
#             marker = markers['aug'] if is_aug else markers['normal']
#
#             # Obtener los valores de exactitud según el modo
#             if mode == "validation":
#                 acc_values = historiales[name]['val_acc']
#                 epochs = range(1, len(acc_values) + 1)
#             else:  # modo "test"
#                 test_acc = test_metrics[name]['test_acc']
#                 acc_values = get_test_acc_by_epoch(historiales[name], test_acc)
#                 epochs = range(1, len(acc_values) + 1)
#
#             # Graficar la curva
#             label = f"{model_type} ({'con aug' if is_aug else 'sin aug'})"
#             ax.plot(epochs, acc_values, linestyle=style, color=color, label=label)
#
#             # Si show_early_stopping es True y tenemos la época donde se habría activado el early stopping
#             if show_early_stopping and 'early_stopping_epoch' in historiales[name]:
#                 es_epoch = historiales[name]['early_stopping_epoch']
#
#                 # Verificamos que la época de early stopping está dentro del rango de las épocas calculadas
#                 if es_epoch <= len(acc_values):
#                     # Obtenemos el valor de exactitud en esa época
#                     # es_value = acc_values[es_epoch - 1]
#
#                     # Marcamos el punto con un marcador
#                     # ax.plot(es_epoch, es_value, marker=marker, color=color,
#                     #         markersize=8, markeredgecolor='black', markeredgewidth=1.5,
#                     #         zorder=10)  # z order alto para asegurar que el marcador esté por encima
#
#                     # Opcionalmente, podemos añadir una línea vertical punteada
#                     ax.axvline(x=es_epoch, color=color, linestyle=':', alpha=0.5)
#
#         ax.set_title(subtitle)
#         ax.set_xlabel('Épocas')
#         ax.set_ylabel(f'Exactitud en {mode} (%)')
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend(loc='lower right')
#
#         # Establecer rango del eje y similar a la figura de referencia
#         ax.set_ylim(0.60, 1.0)
#
#         # Formatear eje y como porcentaje
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x * 100)))
#
#     # Añadimos una nota sobre los marcadores si estamos mostrando puntos de early stopping
#     if show_early_stopping:
#         plt.figtext(0.5, 0.01, "Los marcadores indican el punto donde se habría activado el early stopping",
#                     ha='center', fontsize=9, style='italic')
#
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(os.path.join(vis_results_dir, filename), dpi=300, bbox_inches='tight')
#     plt.show()
#
#     return plt


def plot_optimized_vs_final_comparison(historiales, test_metrics_opt, test_metrics_final, model_names,
                                       title="Comparativa de modelos optimizados vs. finales",
                                       filename="comparativa_opt_final.png",
                                       results_dir='resultados'):
    """
    Genera una visualización comparando el rendimiento de modelos optimizados (con early stopping)
    versus los modelos finales (entrenamiento completo), mostrando los efectos del entrenamiento
    continuado después del punto óptimo.

    Args:
        historiales: Diccionario con historiales de entrenamiento
        test_metrics_opt: Diccionario con métricas de test para modelos optimizados
        test_metrics_final: Diccionario con métricas de test para modelos finales
        model_names: Lista de nombres de modelos a incluir
        title: Título de la gráfica
        filename: Nombre del archivo para guardar la gráfica
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'comparativas')

    # Filtrar los modelos para los que tenemos todos los datos necesarios
    valid_models = [m for m in model_names if m in historiales and m in test_metrics_opt and m in test_metrics_final]

    if not valid_models:
        print("No hay modelos válidos para comparar")
        return

    # Preparamos una figura con subplots para cada modelo
    n_models = len(valid_models)
    fig, axs = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), sharex=False)

    # Si solo hay un modelo, convertimos axs en una lista para mantener consistencia
    if n_models == 1:
        axs = [axs]

    # Colores para los diferentes tipos de curvas
    colors = {
        'train': 'blue',
        'val': 'green',
        'test_opt': 'red',
        'test_final': 'purple'
    }

    for i, model_name in enumerate(valid_models):
        ax = axs[i]
        history = historiales[model_name]

        # Obtenemos el punto de early stopping
        es_epoch = history.get('early_stopping_epoch', len(history['train_acc']))

        # Datos de entrenamiento y validación
        epochs = range(1, len(history['train_acc']) + 1)
        train_acc = history['train_acc']
        val_acc = history['val_acc']

        # Métricas de test
        test_acc_opt = test_metrics_opt[model_name]['test_acc']
        test_acc_final = test_metrics_final[model_name]['test_acc']

        # Graficamos curvas de entrenamiento y validación
        ax.plot(epochs, train_acc, color=colors['train'], label='Entrenamiento')
        ax.plot(epochs, val_acc, color=colors['val'], label='Validación')

        # Marcamos el punto de early stopping con una línea vertical
        ax.axvline(x=es_epoch, color='gray', linestyle='--', alpha=0.7)
        ax.annotate(f"Early stopping\nÉpoca {es_epoch}",
                    xy=(es_epoch, min(train_acc) + 0.01),
                    xytext=(es_epoch + 1, min(train_acc) + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=9)

        # Añadimos líneas horizontales para los valores de test
        ax.axhline(y=test_acc_opt, color=colors['test_opt'], linestyle='-', alpha=0.5)
        ax.axhline(y=test_acc_final, color=colors['test_final'], linestyle='-', alpha=0.5)

        # Añadimos etiquetas para los valores de test
        ax.text(len(epochs) + 1, test_acc_opt, f"Test (opt): {test_acc_opt:.4f}",
                color=colors['test_opt'], verticalalignment='center')
        ax.text(len(epochs) + 1, test_acc_final, f"Test (final): {test_acc_final:.4f}",
                color=colors['test_final'], verticalalignment='center')

        # Sombreamos la región después del early stopping
        ax.axvspan(es_epoch, len(epochs), color='gray', alpha=0.1)

        # Calculamos y mostramos la diferencia entre modelos en test
        diff = (test_acc_final - test_acc_opt) * 100
        diff_color = 'green' if diff >= 0 else 'red'
        diff_text = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"

        y_pos = max(test_acc_opt, test_acc_final) + 0.02
        ax.annotate(f"Δ={diff_text}",
                    xy=(len(epochs) + 1, y_pos),
                    color=diff_color,
                    fontweight='bold')

        # Configuración adicional del subplot
        ax.set_title(f"Modelo: {model_name}")
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Exactitud')
        ax.set_ylim(min(min(train_acc), min(val_acc), test_acc_opt, test_acc_final) - 0.05,
                    max(max(train_acc), max(val_acc), test_acc_opt, test_acc_final) + 0.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

    # Título general y ajuste de layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # Guardamos la figura
    plt.savefig(os.path.join(vis_results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    # También creamos una versión simple de gráfico de barras para comparación directa
    plt.figure(figsize=(12, 6))

    # Preparamos datos para el gráfico de barras
    modelos = valid_models
    acc_opt = [test_metrics_opt[m]['test_acc'] for m in valid_models]
    acc_final = [test_metrics_final[m]['test_acc'] for m in valid_models]

    x = np.arange(len(modelos))
    width = 0.35

    # Creamos el gráfico de barras
    plt.bar(x - width / 2, acc_opt, width, label='Modelo optimizado', color='skyblue')
    plt.bar(x + width / 2, acc_final, width, label='Modelo final', color='salmon')

    # Añadimos etiquetas, título y leyenda
    plt.xlabel('Modelo')
    plt.ylabel('Exactitud en test')
    plt.title('Comparativa directa: Modelos optimizados vs. Modelos finales')
    plt.xticks(x, modelos, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Añadimos las diferencias porcentuales
    for i, (opt, final) in enumerate(zip(acc_opt, acc_final)):
        diff = (final - opt) * 100
        color = 'green' if diff >= 0 else 'red'
        plt.annotate(f"{diff:.2f}%",
                     xy=(float(x[i]), float(max(opt, final) + 0.01)),
                     ha='center', va='bottom',
                     color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, f'barras_{filename}'), dpi=300, bbox_inches='tight')
    plt.show()

    return plt