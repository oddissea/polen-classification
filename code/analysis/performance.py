# ************************************************************************
# * analysis/performance.py
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
Módulo para análisis y visualización del rendimiento de modelos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from .visualization import configure_output_dir


def generar_tabla_comparativa(historiales, nombres_modelos, results_dir='resultados'):
    """
    Genera una tabla comparativa con los mejores resultados de cada modelo.
    Muestra la tabla en formato texto y la guarda como CSV.

    Args:
        historiales: Diccionario con historiales de entrenamiento
        nombres_modelos: Lista de nombres de modelos a incluir
        results_dir: Directorio donde guardar los resultados
    """

    vis_results_dir = configure_output_dir(results_dir, 'performance')

    # Creamos los datos para la tabla
    datos_tabla = []
    columnas = ['Modelo', 'Exactitud Train', 'Exactitud Val', 'Diferencia', 'Mejor Época']

    for nombre in nombres_modelos:
        if nombre in historiales:
            hist = historiales[nombre]
            mejor_val_acc = max(hist['val_acc'])
            epoca_mejor = hist['val_acc'].index(mejor_val_acc) + 1
            train_acc = hist['train_acc'][epoca_mejor - 1]
            diferencia = train_acc - mejor_val_acc

            datos_tabla.append([
                nombre,
                f"{train_acc:.4f}",
                f"{mejor_val_acc:.4f}",
                f"{diferencia:.4f}",
                f"{epoca_mejor}"
            ])

    # Guardamos la tabla como CSV
    df = pd.DataFrame(datos_tabla, columns=columnas)
    csv_path = os.path.join(vis_results_dir, 'tabla_comparativa.csv')
    df.to_csv(csv_path, index=False)
    print(f"Tabla comparativa guardada en CSV: {csv_path}")

    # Mostramos la tabla en formato texto con formato mejorado
    print("\n=== Tabla comparativa de modelos ===")

    # Determinamos el ancho máximo de cada columna para un formato más limpio
    anchos = [max(len(str(fila[i])) for fila in datos_tabla + [columnas]) for i in range(len(columnas))]
    anchos = [max(ancho + 2, 12) for ancho in anchos]  # Aseguramos un mínimo de 12 caracteres

    # Imprimimos el encabezado
    header = " | ".join(f"{col:{anchos[i]}s}" for i, col in enumerate(columnas))
    print(header)
    print("-" * len(header))

    # Imprimimos cada fila
    for fila in datos_tabla:
        print(" | ".join(f"{str(val):{anchos[i]}s}" for i, val in enumerate(fila)))
    print("")

def _visualizar_matriz_confusion(ax, matriz, matriz_norm, titulo, cmap='Blues'):
    """
    Función auxiliar para visualizar una matriz de confusión en un eje específico.

    Args:
        :arg ax: El eje (subplot) donde dibujar
        :arg matriz: la matriz de confusión original
        :arg matriz_norm: La matriz de confusión normalizada
        :arg titulo: Título para el gráfico
        :arg cmap: Mapa de colores a utilizar
    """
    ax.imshow(matriz_norm, cmap=cmap)
    ax.set_title(titulo)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Kunzea', 'Lepto'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Kunzea', 'Lepto'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')

    # Añadimos los valores a la matriz
    for y in range(matriz.shape[0]):
        for x in range(matriz.shape[1]):
            ax.text(x, y, f"{matriz[y, x]}\n({matriz_norm[y, x]:.2f})",
                    ha="center", va="center",
                    color="black" if matriz_norm[y, x] < 0.5 else "white")


def _comparar_matrices_confusion(matriz_a, matriz_b, titulo_a, titulo_b, titulo_diff, prefix, results_dir):
    """
    Función auxiliar para comparar dos matrices de confusión y visualizar sus diferencias.

    Args:
        matriz_a: Primera matriz de confusión
        matriz_b: Segunda matriz de confusión
        titulo_a: Título para la primera matriz
        titulo_b: Título para la segunda matriz
        titulo_diff: Título para la matriz de diferencia
        prefix: Prefijo para los nombres de archivo
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'matrices_confusion')

    # Calculamos la diferencia entre matrices
    diff_matrix = matriz_b - matriz_a

    # Normalizamos las matrices para mejor comparación
    total_a = np.sum(matriz_a)
    matriz_a_norm = matriz_a / total_a

    total_b = np.sum(matriz_b)
    matriz_b_norm = matriz_b / total_b

    # Creamos una nueva figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Primera matriz
    _visualizar_matriz_confusion(
        axes[0], matriz_a, matriz_a_norm,
        titulo_a, cmap='Blues'
    )

    # Segunda matriz
    _visualizar_matriz_confusion(
        axes[1], matriz_b, matriz_b_norm,
        titulo_b, cmap='Blues'
    )

    # Diferencia entre matrices
    axes[2].imshow(diff_matrix, cmap='RdBu', vmin=-np.max(abs(diff_matrix)), vmax=np.max(abs(diff_matrix)))
    axes[2].set_title(titulo_diff)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Kunzea', 'Lepto'])
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Kunzea', 'Lepto'])
    axes[2].set_xlabel('Predicción')
    axes[2].set_ylabel('Real')

    # Añadimos los valores a la matriz de diferencia
    for y in range(diff_matrix.shape[0]):
        for x in range(diff_matrix.shape[1]):
            axes[2].text(x, y, f"{diff_matrix[y, x]}",
                         ha="center", va="center",
                         color="black" if abs(diff_matrix[y, x]) < 0.5 * np.max(abs(diff_matrix)) else "white")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, f'{prefix}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Convertimos matrices a DataFrames
    df_a = pd.DataFrame(matriz_a,
                        index=['Kunzea Real', 'Lepto Real'],
                        columns=['Kunzea Pred', 'Lepto Pred'])
    df_b = pd.DataFrame(matriz_b,
                        index=['Kunzea Real', 'Lepto Real'],
                        columns=['Kunzea Pred', 'Lepto Pred'])
    df_diff = pd.DataFrame(diff_matrix,
                           index=['Kunzea Real', 'Lepto Real'],
                           columns=['Kunzea Pred', 'Lepto Pred'])

    # Guardamos en CSV
    df_a.to_csv(os.path.join(vis_results_dir, f'{prefix}_a.csv'))
    df_b.to_csv(os.path.join(vis_results_dir, f'{prefix}_b.csv'))
    df_diff.to_csv(os.path.join(vis_results_dir, f'{prefix}_diff.csv'))

    # Devolvemos los componentes de las matrices para análisis adicional
    return matriz_a.ravel(), matriz_b.ravel()


def analizar_matrices_confusion(test_metrics, models_base, results_dir='resultados', test_metrics_final=None):
    """
    Compara visualmente las matrices de confusión de modelos.
    Si test_metrics_final se proporciona, compara modelos optimizados vs. finales.
    De lo contrario, compara modelos con y sin data augmentation.
    """
    # Análisis de data augmentation
    vis_dir_normal_aug = configure_output_dir(results_dir, 'matrices_confusion')
    print("\n--- Análisis de matrices de confusión: Normal vs Augmentation ---")

    for modelo_base in models_base:
        normal_key = f"{modelo_base}_normal"
        aug_key = f"{modelo_base}_aug"

        if normal_key in test_metrics and aug_key in test_metrics:
            # Obtenemos las matrices de confusión
            conf_matrix_normal = test_metrics[normal_key]['conf_matrix']
            conf_matrix_aug = test_metrics[aug_key]['conf_matrix']

            # Usamos la función auxiliar para la comparación
            (tn_normal, fp_normal, fn_normal, tp_normal), (
            tn_aug, fp_aug, fn_aug, tp_aug) = _comparar_matrices_confusion(
                conf_matrix_normal, conf_matrix_aug,
                f"{modelo_base} sin aug", f"{modelo_base} con aug",
                "Diferencia (Aug - Normal)",
                f'matriz_confusion_{modelo_base}',
                vis_dir_normal_aug
            )

            # Imprimimos análisis textual
            print(f"\nModelo: {modelo_base}")
            print(f"Matriz de confusión sin augmentation:")
            print(conf_matrix_normal)
            print(f"Matriz de confusión con augmentation:")
            print(conf_matrix_aug)

            # Calculamos mejoras en cada clase
            mejora_kunzea = (tn_aug / (tn_aug + fp_aug)) - (tn_normal / (tn_normal + fp_normal))
            mejora_lepto = (tp_aug / (tp_aug + fn_aug)) - (tp_normal / (tp_normal + fn_normal))

            print(f"Mejora en exactitud de Kunzea (clase 0): {mejora_kunzea:.4f} ({mejora_kunzea * 100:.2f}%)")
            print(f"Mejora en exactitud de Lepto (clase 1): {mejora_lepto:.4f} ({mejora_lepto * 100:.2f}%)")

            # Evaluamos si el modelo es más equilibrado
            desequilibrio_normal = abs((tn_normal / (tn_normal + fp_normal)) - (tp_normal / (tp_normal + fn_normal)))
            desequilibrio_aug = abs((tn_aug / (tn_aug + fp_aug)) - (tp_aug / (tp_aug + fn_aug)))

            print(f"Desequilibrio entre clases sin augmentation: {desequilibrio_normal:.4f}")
            print(f"Desequilibrio entre clases con augmentation: {desequilibrio_aug:.4f}")

            if desequilibrio_aug < desequilibrio_normal:
                print(
                    f"El data augmentation ha mejorado el equilibrio entre clases en un {(1 - desequilibrio_aug / desequilibrio_normal) * 100:.2f}%")
            else:
                print(
                    f"El data augmentation ha aumentado el desequilibrio entre clases en un {(desequilibrio_aug / desequilibrio_normal - 1) * 100:.2f}%")

    # Análisis de modelos optimizados vs. finales
    if test_metrics_final:
        vis_dir_opt_final = configure_output_dir(results_dir, 'matrices_confusion_opt_final')
        print("\n--- Análisis de matrices de confusión: Optimizado vs Final ---")

        for modelo_key in test_metrics.keys():
            if modelo_key in test_metrics_final:
                # Obtenemos las matrices de confusión
                conf_matrix_opt = test_metrics[modelo_key]['conf_matrix']
                conf_matrix_final = test_metrics_final[modelo_key]['conf_matrix']

                # Usamos la función auxiliar para la comparación
                (tn_opt, fp_opt, fn_opt, tp_opt), (
                tn_final, fp_final, fn_final, tp_final) = _comparar_matrices_confusion(
                    conf_matrix_opt, conf_matrix_final,
                    f"{modelo_key} optimizado", f"{modelo_key} final",
                    "Diferencia (Final - Optimizado)",
                    f'matriz_confusion_opt_final_{modelo_key}',
                    vis_dir_opt_final
                )

                # Análisis textual
                print(f"\nModelo: {modelo_key}")
                print(
                    f"Cambios en clasificación Kunzea (TN): {tn_final - tn_opt} ({(tn_final - tn_opt) / tn_opt * 100:.2f}%)")
                print(
                    f"Cambios en clasificación Lepto (TP): {tp_final - tp_opt} ({(tp_final - tp_opt) / tp_opt * 100:.2f}%)")

def analizar_metricas_por_clase(test_metrics, models_base, results_dir='resultados'):
    """
    Analiza y compara métricas específicas por clase para modelos con y sin data augmentation.

    Args:
        test_metrics: Diccionario con métricas de test
        models_base: Lista de nombres base de modelos (sin sufijos '_normal'/'_aug')
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'metricas')
    print("\n--- Análisis de métricas por clase ---")

    # Preparamos datos para una gráfica de barras comparativa
    modelos = []
    precision_kunzea_normal = []
    precision_kunzea_aug = []
    precision_lepto_normal = []
    precision_lepto_aug = []
    recall_kunzea_normal = []
    recall_kunzea_aug = []
    recall_lepto_normal = []
    recall_lepto_aug = []

    for modelo_base in models_base:
        normal_key = f"{modelo_base}_normal"
        aug_key = f"{modelo_base}_aug"

        if normal_key in test_metrics and aug_key in test_metrics:
            modelos.append(modelo_base)

            # Extraemos datos de las matrices de confusión
            conf_matrix_normal = test_metrics[normal_key]['conf_matrix']
            conf_matrix_aug = test_metrics[aug_key]['conf_matrix']

            # Calculamos precision y recall por clase
            tn_normal, fp_normal, fn_normal, tp_normal = conf_matrix_normal.ravel()
            tn_aug, fp_aug, fn_aug, tp_aug = conf_matrix_aug.ravel()

            # Precision por clase
            # Clase 0 (Kunzea): TN / (TN + FN)
            prec_kunzea_normal = tn_normal / (tn_normal + fn_normal) if (tn_normal + fn_normal) > 0 else 0
            prec_kunzea_aug = tn_aug / (tn_aug + fn_aug) if (tn_aug + fn_aug) > 0 else 0

            # Clase 1 (Lepto): TP / (TP + FP)
            prec_lepto_normal = tp_normal / (tp_normal + fp_normal) if (tp_normal + fp_normal) > 0 else 0
            prec_lepto_aug = tp_aug / (tp_aug + fp_aug) if (tp_aug + fp_aug) > 0 else 0

            # Recall por clase
            # Clase 0 (Kunzea): TN / (TN + FP)
            rec_kunzea_normal = tn_normal / (tn_normal + fp_normal) if (tn_normal + fp_normal) > 0 else 0
            rec_kunzea_aug = tn_aug / (tn_aug + fp_aug) if (tn_aug + fp_aug) > 0 else 0

            # Clase 1 (Lepto): TP / (TP + FN)
            rec_lepto_normal = tp_normal / (tp_normal + fn_normal) if (tp_normal + fn_normal) > 0 else 0
            rec_lepto_aug = tp_aug / (tp_aug + fn_aug) if (tp_aug + fn_aug) > 0 else 0

            # Guardamos valores
            precision_kunzea_normal.append(prec_kunzea_normal)
            precision_kunzea_aug.append(prec_kunzea_aug)
            precision_lepto_normal.append(prec_lepto_normal)
            precision_lepto_aug.append(prec_lepto_aug)
            recall_kunzea_normal.append(rec_kunzea_normal)
            recall_kunzea_aug.append(rec_kunzea_aug)
            recall_lepto_normal.append(rec_lepto_normal)
            recall_lepto_aug.append(rec_lepto_aug)

            # Imprimimos análisis textual
            print(f"\nModelo: {modelo_base}")
            print(
                f"Precision Kunzea - Sin aug: {prec_kunzea_normal:.4f}, Con aug: {prec_kunzea_aug:.4f}, Diferencia: {prec_kunzea_aug - prec_kunzea_normal:.4f}")
            print(
                f"Precision Lepto - Sin aug: {prec_lepto_normal:.4f}, Con aug: {prec_lepto_aug:.4f}, Diferencia: {prec_lepto_aug - prec_lepto_normal:.4f}")
            print(
                f"Recall Kunzea - Sin aug: {rec_kunzea_normal:.4f}, Con aug: {rec_kunzea_aug:.4f}, Diferencia: {rec_kunzea_aug - rec_kunzea_normal:.4f}")
            print(
                f"Recall Lepto - Sin aug: {rec_lepto_normal:.4f}, Con aug: {rec_lepto_aug:.4f}, Diferencia: {rec_lepto_aug - rec_lepto_normal:.4f}")

            # Análisis de equilibrio
            equilibrio_prec_normal = abs(prec_kunzea_normal - prec_lepto_normal)
            equilibrio_prec_aug = abs(prec_kunzea_aug - prec_lepto_aug)
            equilibrio_rec_normal = abs(rec_kunzea_normal - rec_lepto_normal)
            equilibrio_rec_aug = abs(rec_kunzea_aug - rec_lepto_aug)

            print(
                f"Desequilibrio en precision sin aug: {equilibrio_prec_normal:.4f}, con aug: {equilibrio_prec_aug:.4f}")
            print(f"Desequilibrio en recall sin aug: {equilibrio_rec_normal:.4f}, con aug: {equilibrio_rec_aug:.4f}")

    # Creamos gráficas comparativas si tenemos datos
    if modelos:
        # Creamos un DataFrame para precision
        precision_data = {
            'Modelo': modelos,
            'Kunzea_Normal': precision_kunzea_normal,
            'Kunzea_Aug': precision_kunzea_aug,
            'Lepto_Normal': precision_lepto_normal,
            'Lepto_Aug': precision_lepto_aug,
            'Diff_Kunzea': [aug - norm for aug, norm in zip(precision_kunzea_aug, precision_kunzea_normal)],
            'Diff_Lepto': [aug - norm for aug, norm in zip(precision_lepto_aug, precision_lepto_normal)]
        }

        precision_df = pd.DataFrame(precision_data)
        precision_csv_path = os.path.join(vis_results_dir, 'precision_por_clase.csv')
        precision_df.to_csv(precision_csv_path, index=False)
        print(f"Métricas de precisión guardadas en: {precision_csv_path}")

        # Creamos un DataFrame para recall
        recall_data = {
            'Modelo': modelos,
            'Kunzea_Normal': recall_kunzea_normal,
            'Kunzea_Aug': recall_kunzea_aug,
            'Lepto_Normal': recall_lepto_normal,
            'Lepto_Aug': recall_lepto_aug,
            'Diff_Kunzea': [aug - norm for aug, norm in zip(recall_kunzea_aug, recall_kunzea_normal)],
            'Diff_Lepto': [aug - norm for aug, norm in zip(recall_lepto_aug, recall_lepto_normal)]
        }

        recall_df = pd.DataFrame(recall_data)
        recall_csv_path = os.path.join(vis_results_dir, 'recall_por_clase.csv')
        recall_df.to_csv(recall_csv_path, index=False)
        print(f"Métricas de recall guardadas en: {recall_csv_path}")

        # Gráfica de precision
        plt.figure(figsize=(12, 6))
        x = np.arange(len(modelos))
        width = 0.2

        plt.bar(x - 1.5 * width, precision_kunzea_normal, width, label='Kunzea - Sin Aug')
        plt.bar(x - 0.5 * width, precision_kunzea_aug, width, label='Kunzea - Con Aug')
        plt.bar(x + 0.5 * width, precision_lepto_normal, width, label='Lepto - Sin Aug')
        plt.bar(x + 1.5 * width, precision_lepto_aug, width, label='Lepto - Con Aug')

        plt.xlabel('Modelo')
        plt.ylabel('Precision')
        plt.title('Comparativa de Precision por clase')
        plt.xticks(x, modelos)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_results_dir, 'comparacion_precision_por_clase.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Gráfica de recall
        plt.figure(figsize=(12, 6))

        plt.bar(x - 1.5 * width, recall_kunzea_normal, width, label='Kunzea - Sin Aug')
        plt.bar(x - 0.5 * width, recall_kunzea_aug, width, label='Kunzea - Con Aug')
        plt.bar(x + 0.5 * width, recall_lepto_normal, width, label='Lepto - Sin Aug')
        plt.bar(x + 1.5 * width, recall_lepto_aug, width, label='Lepto - Con Aug')

        plt.xlabel('Modelo')
        plt.ylabel('Recall')
        plt.title('Comparativa de Recall por clase')
        plt.xticks(x, modelos)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_results_dir, 'comparacion_recall_por_clase.png'), dpi=300, bbox_inches='tight')
        plt.show()


def comparar_optimizados_vs_finales(test_metrics_opt, test_metrics_final, models_base, historiales, results_dir='resultados'):
    """
    Compara el rendimiento de modelos optimizados (early stopping) vs. modelos finales (entrenamiento completo).

    Args:
        historiales: Diccionario con historiales de entrenamiento
        test_metrics_opt: Diccionario con métricas de test para modelos optimizados
        test_metrics_final: Diccionario con métricas de test para modelos finales
        models_base: Lista de nombres base de modelos (sin sufijos)
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'comparativa_opt_final')
    print("\n--- Análisis comparativo: Modelos optimizados vs. Modelos finales ---")

    # Preparamos datos para gráficas y tablas
    modelos = []
    acc_opt = []
    acc_final = []
    f1_opt = []
    f1_final = []
    diferencias = []

    for modelo_base in models_base:
        # Para cada variante (normal y aug)
        for suffix in ['normal', 'aug']:
            modelo_key = f"{modelo_base}_{suffix}"
            if modelo_key in test_metrics_opt and modelo_key in test_metrics_final:
                modelos.append(modelo_key)

                # Extraemos métricas
                acc_opt_val = test_metrics_opt[modelo_key]['test_acc']
                acc_final_val = test_metrics_final[modelo_key]['test_acc']
                f1_opt_val = test_metrics_opt[modelo_key]['f1']
                f1_final_val = test_metrics_final[modelo_key]['f1']

                # Calculamos diferencia
                diff = (acc_final_val - acc_opt_val) * 100  # En porcentaje

                # Guardamos valores
                acc_opt.append(acc_opt_val)
                acc_final.append(acc_final_val)
                f1_opt.append(f1_opt_val)
                f1_final.append(f1_final_val)
                diferencias.append(diff)

                # Imprimimos análisis textual
                print(f"{modelo_key}: Optimizado {acc_opt_val:.4f} vs Final {acc_final_val:.4f} (Diff: {diff:.2f}%)")

                # Análisis de posible sobre ajuste
                train_val_diff_opt = abs(
                    historiales[modelo_key]['train_acc'][test_metrics_opt[modelo_key].get('stopped_epoch', -1)] -
                    historiales[modelo_key]['val_acc'][test_metrics_opt[modelo_key].get('stopped_epoch', -1)])

                train_val_diff_final = abs(historiales[modelo_key]['train_acc'][-1] -
                                           historiales[modelo_key]['val_acc'][-1])

                # Determinamos si hay overfitting
                if diff < -1.0 and train_val_diff_final > train_val_diff_opt:
                    print(
                        f"  ⚠️ Posible sobre ajuste detectado: la diferencia train-val aumenta de {train_val_diff_opt:.4f} a {train_val_diff_final:.4f}")

    # Creamos una gráfica de barras comparativa
    if modelos:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(modelos))
        width = 0.35

        plt.bar(x - width / 2, acc_opt, width, label='Modelo optimizado')
        plt.bar(x + width / 2, acc_final, width, label='Modelo final')

        plt.ylabel('Exactitud en test')
        plt.title('Comparativa de rendimiento: Modelos optimizados vs. Modelos finales')
        plt.xticks(x, modelos, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Añadimos valores de diferencia
        # Corregimos el problema con las anotaciones:
        for i, diff in enumerate(diferencias):
            color = 'green' if diff >= 0 else 'red'
            plt.annotate(f"{diff:.2f}%",
                         xy=(float(x[i]), float(max(acc_opt[i], acc_final[i]) + 0.01)),  # Convertimos a float
                         ha='center', va='bottom',
                         color=color, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_results_dir, 'comparativa_opt_final.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Creamos un DataFrame para guardar los resultados
        comparativa_data = {
            'Modelo': modelos,
            'Exactitud_Optimizado': acc_opt,
            'Exactitud_Final': acc_final,
            'Diferencia_Porcentual': diferencias,
            'F1_Optimizado': f1_opt,
            'F1_Final': f1_final
        }

        df = pd.DataFrame(comparativa_data)
        csv_path = os.path.join(vis_results_dir, 'comparativa_opt_final.csv')
        df.to_csv(csv_path, index=False)
        print(f"Comparativa guardada en: {csv_path}")


def generar_tabla_comparativa_opt_final(test_metrics_opt, test_metrics_final, models_base, results_dir='resultados'):
    """
    Genera una tabla comparativa detallada entre modelos optimizados y finales.

    Args:
        test_metrics_opt: Diccionario con métricas de test para modelos optimizados
        test_metrics_final: Diccionario con métricas de test para modelos finales
        models_base: Lista de nombres base de modelos (sin sufijos)
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'tablas_comparativas')

    # Preparamos los datos para la tabla
    filas = []

    for modelo_base in models_base:
        for suffix in ['normal', 'aug']:
            modelo_key = f"{modelo_base}_{suffix}"
            if modelo_key in test_metrics_opt and modelo_key in test_metrics_final:
                # Métricas del modelo optimizado
                opt_acc = test_metrics_opt[modelo_key]['test_acc']
                opt_prec = test_metrics_opt[modelo_key]['precision']
                opt_recall = test_metrics_opt[modelo_key]['recall']
                opt_f1 = test_metrics_opt[modelo_key]['f1']

                # Métricas del modelo final
                final_acc = test_metrics_final[modelo_key]['test_acc']
                final_prec = test_metrics_final[modelo_key]['precision']
                final_recall = test_metrics_final[modelo_key]['recall']
                final_f1 = test_metrics_final[modelo_key]['f1']

                # Diferencias (en porcentaje)
                diff_acc = (final_acc - opt_acc) * 100
                diff_f1 = (final_f1 - opt_f1) * 100

                # Agregamos a las filas de la tabla
                fila = {
                    'Modelo': modelo_key,
                    'Test_Acc_Opt': opt_acc,
                    'Test_Acc_Final': final_acc,
                    'Diff_Acc': diff_acc,
                    'F1_Opt': opt_f1,
                    'F1_Final': final_f1,
                    'Diff_F1': diff_f1,
                    'Precision_Opt': opt_prec,
                    'Precision_Final': final_prec,
                    'Recall_Opt': opt_recall,
                    'Recall_Final': final_recall
                }
                filas.append(fila)

    # Creamos y guardamos el DataFrame
    if filas:
        df = pd.DataFrame(filas)
        csv_path = os.path.join(vis_results_dir, 'tabla_comparativa_opt_final.csv')
        df.to_csv(csv_path, index=False)
        print(f"Tabla comparativa guardada en: {csv_path}")

        # Mostramos un resumen textual
        print("\n=== Resumen comparativo entre modelos optimizados y finales ===")
        print(f"{'Modelo':<20} {'Opt Acc':<10} {'Final Acc':<10} {'Diff':<10}")
        print("-" * 50)

        for fila in filas:
            print(
                f"{fila['Modelo']:<20} {fila['Test_Acc_Opt']:.4f}     {fila['Test_Acc_Final']:.4f}     {fila['Diff_Acc']:.2f}%")

def analizar_activaciones_modelos_basicos(model, test_loader, modelo_nombre='net1', results_dir='resultados'):
    """
    Analiza las activaciones de los modelos básicos para diferentes clases.

    Args:
        model: Modelo entrenado (NET1 o NET2)
        test_loader: DataLoader del conjunto de test
        modelo_nombre: Nombre del modelo para guardar resultados
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'activaciones')
    device = next(model.parameters()).device
    model.eval()

    # Recopilamos activaciones por clase
    activaciones_kunzea = []
    activaciones_lepto = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Obtenemos predicciones
            outputs = model(inputs)

            # Almacenamos activaciones por clase
            for index, label in enumerate(labels):
                if label.item() == 0:  # Kunzea
                    activaciones_kunzea.append(outputs[index].item())
                else:  # Lepto
                    activaciones_lepto.append(outputs[index].item())

    # Convertimos a numpy para análisis
    activaciones_kunzea = np.array(activaciones_kunzea)
    activaciones_lepto = np.array(activaciones_lepto)

    # Visualizamos distribución de activaciones
    plt.figure(figsize=(10, 6))

    plt.hist(activaciones_kunzea, bins=20, alpha=0.5, label='Kunzea')
    plt.hist(activaciones_lepto, bins=20, alpha=0.5, label='Lepto')

    plt.xlabel('Activación (probabilidad predicha)')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de activaciones por clase - {modelo_nombre}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, f'activaciones_{modelo_nombre}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Análisis estadístico
    print(f"Estadísticas de activaciones para {modelo_nombre}:")
    print(f"Kunzea - Media: {activaciones_kunzea.mean():.4f}, Desv. Est.: {activaciones_kunzea.std():.4f}")
    print(f"Lepto - Media: {activaciones_lepto.mean():.4f}, Desv. Est.: {activaciones_lepto.std():.4f}")
    print(f"Diferencia de medias: {np.abs(activaciones_kunzea.mean() - activaciones_lepto.mean()):.4f}")

    # Calcular separabilidad
    umbral = 0.5
    kunzea_bien_clasificados = np.sum(activaciones_kunzea < umbral) / len(activaciones_kunzea)
    lepto_bien_clasificados = np.sum(activaciones_lepto >= umbral) / len(activaciones_lepto)

    print(f"Precisión para Kunzea: {kunzea_bien_clasificados:.4f}")
    print(f"Precisión para Lepto: {lepto_bien_clasificados:.4f}")
    print(f"Precisión promedio: {(kunzea_bien_clasificados + lepto_bien_clasificados) / 2:.4f}")

    # Creamos un DataFrame con las estadísticas generales
    stats_df = pd.DataFrame({
        'Modelo': [modelo_nombre],
        'Media_Kunzea': [activaciones_kunzea.mean()],
        'Desv_Est_Kunzea': [activaciones_kunzea.std()],
        'Media_Lepto': [activaciones_lepto.mean()],
        'Desv_Est_Lepto': [activaciones_lepto.std()],
        'Diferencia_Medias': [np.abs(activaciones_kunzea.mean() - activaciones_lepto.mean())],
        'Precision_Kunzea': [kunzea_bien_clasificados],
        'Precision_Lepto': [lepto_bien_clasificados],
        'Precision_Promedio': [(kunzea_bien_clasificados + lepto_bien_clasificados) / 2]
    })

    # Guardamos las estadísticas
    stats_csv_path = os.path.join(vis_results_dir, f'statistics_activaciones_{modelo_nombre}.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Estadísticas de activaciones guardadas en: {stats_csv_path}")

    # Opcionalmente, podemos guardar también los valores individuales de activación
    # Esto puede ser útil para análisis más detallados
    activaciones_df = pd.DataFrame({
        'Clase': ['Kunzea'] * len(activaciones_kunzea) + ['Lepto'] * len(activaciones_lepto),
        'Activacion': np.concatenate([activaciones_kunzea, activaciones_lepto])
    })

    act_csv_path = os.path.join(vis_results_dir, f'valores_activaciones_{modelo_nombre}.csv')
    activaciones_df.to_csv(act_csv_path, index=False)
    print(f"Valores de activaciones guardados en: {act_csv_path}")

def generar_tabla_resumen_modelos(metricas_net1, metricas_net2, metricas_cnn, test_metrics, results_dir='resultados'):
    """
    Genera una tabla de resumen que compara todos los modelos,
    incluyendo métricas de los pesos y rendimiento de clasificación.

    Args:
        metricas_net1: Métricas de pesos para modelos NET1
        metricas_net2: Métricas de pesos para modelos NET2
        metricas_cnn: Métricas de pesos para modelos CNN
        test_metrics: Métricas de rendimiento en test para todos los modelos
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'resumen')

    # Columnas para la tabla de resumen
    columnas = ['Modelo', 'Data Aug.', 'Test Acc.', 'Arch.', 'Parámetros', 'Media Pesos', 'Pesos ≈ 0', 'Tiempo (s)']

    # Preparamos los datos
    datos_tabla = []

    # Agregamos datos para NET1
    for model_name, metricas in metricas_net1.items():
        # Determinamos si es versión con augmentation
        con_aug = "aug" in model_name.lower()

        # Obtenemos métricas de test
        test_acc = test_metrics.get(model_name, {}).get('test_acc', 'N/A')
        if isinstance(test_acc, float):
            test_acc = f"{test_acc:.4f}"

        fila = [
            model_name,
            "Sí" if con_aug else "No",
            test_acc,
            "1L-FC",  # Arquitectura: 1 capa fully connected
            f"{metricas['num_pesos']}",
            f"{metricas['media']:.4f}",
            f"{metricas['pct_cerca_cero']:.2f}%",
            # Tiempo de entrenamiento - habría que extraerlo de los historiales
            "—"
        ]
        datos_tabla.append(fila)

    # Agregamos datos para NET2
    for model_name, metricas in metricas_net2.items():
        # Sólo tomamos las métricas globales
        global_metrics = metricas['global']

        # Determinamos si es versión con augmentation
        con_aug = "aug" in model_name.lower()

        # Obtenemos métricas de test
        test_acc = test_metrics.get(model_name, {}).get('test_acc', 'N/A')
        if isinstance(test_acc, float):
            test_acc = f"{test_acc:.4f}"

        fila = [
            model_name,
            "Sí" if con_aug else "No",
            test_acc,
            "MLP",  # Arquitectura: MLP con capas ocultas
            f"{global_metrics['num_pesos']}",
            f"{global_metrics['media']:.4f}",
            f"{global_metrics['pct_cerca_cero']:.2f}%",
            # Tiempo de entrenamiento
            "—"
        ]
        datos_tabla.append(fila)

    # Agregamos datos para CNN
    for model_name, metricas in metricas_cnn.items():
        # Determinamos si es versión con augmentation
        con_aug = "aug" in model_name.lower()

        # Obtenemos métricas de test
        test_acc = test_metrics.get(model_name, {}).get('test_acc', 'N/A')
        if isinstance(test_acc, float):
            test_acc = f"{test_acc:.4f}"

        # Contamos el número total de parámetros
        total_params = sum(capa['num_parametros'] for capa in metricas['capas'])

        # Calculamos la media ponderada de pesos cercanos a cero
        pct_cerca_cero = np.mean([capa['pct_cerca_cero'] for capa in metricas['capas']])

        # Calculamos la media ponderada de los valores de pesos
        media_pesos = np.mean([capa['media'] for capa in metricas['capas']])

        fila = [
            model_name,
            "Sí" if con_aug else "No",
            test_acc,
            "CNN",  # Arquitectura: CNN
            f"{total_params}",
            f"{media_pesos:.4f}",
            f"{pct_cerca_cero:.2f}%",
            # Tiempo de entrenamiento
            "—"
        ]
        datos_tabla.append(fila)

    # Ordenamos la tabla primero por tipo de arquitectura, luego por augmentation
    datos_tabla.sort(key=lambda x: (x[3], x[1]))

    # También mostramos la tabla en formato texto
    print("\n=== Resumen comparativo de todos los modelos ===")
    # Imprimimos encabezados
    cabecera = " | ".join(columnas)
    print(cabecera)
    print("-" * len(cabecera))
    # Imprimimos filas
    for fila in datos_tabla:
        print(" | ".join(fila))

    # CSV
    df = pd.DataFrame(datos_tabla, columns=columnas)

    # Guardamos como CSV en el subdirectorio de resúmenes
    csv_path = os.path.join(vis_results_dir, "resumen_modelos.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tabla de resumen guardada en CSV: {csv_path}")