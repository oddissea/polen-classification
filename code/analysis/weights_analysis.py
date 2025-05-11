# ************************************************************************
# * analysis/weights_analysis.py
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
Módulo para el análisis de pesos y arquitectura de modelos.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from .visualization import configure_output_dir


def extraer_metricas_pesos(pesos, nombre='modelo'):
    """
    Extrae métricas estadísticas de los pesos y devuelve un diccionario con los resultados.

    Args:
        pesos: Tensor o array de pesos
        nombre: Nombre identificativo para las métricas

    Returns:
        dict: Diccionario con métricas estadísticas
    """
    # Convertimos a numpy si es un tensor
    if torch.is_tensor(pesos):
        pesos = pesos.detach().cpu().numpy()

    # Aplanamos el array para análisis estadístico
    pesos_flat = pesos.flatten()

    # Calculamos métricas estadísticas básicas
    metricas = {
        'modelo': nombre,
        'media': float(np.mean(pesos_flat)),
        'mediana': float(np.median(pesos_flat)),
        'desv_std': float(np.std(pesos_flat)),
        'min': float(np.min(pesos_flat)),
        'max': float(np.max(pesos_flat)),
        'rango': float(np.max(pesos_flat) - np.min(pesos_flat)),
        'num_pesos': int(pesos_flat.size),
    }

    # Calculamos métricas de dispersión
    umbral_cero = 0.01  # Pesos cercanos a cero (en valor absoluto)
    pesos_cerca_cero = np.sum(np.abs(pesos_flat) < umbral_cero)
    metricas['pct_cerca_cero'] = float(pesos_cerca_cero / pesos_flat.size * 100)

    # Distribución de pesos positivos y negativos
    metricas['pct_positivos'] = float(np.sum(pesos_flat > 0) / pesos_flat.size * 100)
    metricas['pct_negativos'] = float(np.sum(pesos_flat < 0) / pesos_flat.size * 100)

    return metricas


def visualizar_pesos_conv(model, capa_idx=0, nombre_modelo='cnn', num_filtros=16, results_dir='resultados'):
    """
    Visualiza los pesos (kernels) de una capa convolucional específica.

    Args:
        model: Modelo CNN entrenado
        capa_idx: Índice de la capa convolucional a visualizar
        nombre_modelo: Nombre del modelo para guardar visualización
        num_filtros: Número máximo de filtros a visualizar
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'pesos')

    # Obtenemos todas las capas convolucionales
    capas_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    if capa_idx >= len(capas_conv):
        print(f"Error: El model solo tiene {len(capas_conv)} capas convolucionales")
        return

    # Obtenemos los pesos de la capa especificada
    pesos = capas_conv[capa_idx].weight.data.cpu().numpy()

    # Limitamos el número de filtros a visualizar
    num_filtros = min(num_filtros, pesos.shape[0])

    # Determinamos el número de filas y columnas para la visualización
    n_filas = int(np.ceil(np.sqrt(num_filtros)))
    n_cols = int(np.ceil(num_filtros / n_filas))

    # Creamos la figura
    plt.figure(figsize=(12, 12))

    # Normalizamos los valores para mejor visualización
    min_val = np.min(pesos)
    max_val = np.max(pesos)

    # Visualizamos cada filtro
    for index in range(num_filtros):
        plt.subplot(n_filas, n_cols, index + 1)

        # Para filtros de la primera capa (que reciben 1 canal)
        if pesos.shape[1] == 1:
            img = pesos[index, 0]
        else:
            # Para filtros de capas intermedias (promedio de canales)
            img = np.mean(pesos[index], axis=0)

        # Normalizamos para visualización
        img = (img - min_val) / (max_val - min_val + 1e-8)

        plt.imshow(img, cmap='viridis')
        plt.title(f'Filtro {index + 1}')
        plt.axis('off')

    plt.suptitle(f'Filtros de la capa convolucional {capa_idx + 1} del model {nombre_modelo}')
    plt.tight_layout()

    # Guardamos la visualización
    plt.savefig(os.path.join(vis_results_dir, f'pesos_conv{capa_idx + 1}_{nombre_modelo}.png'), dpi=300,
                bbox_inches='tight')
    plt.show()


def visualizar_pesos_net1(model, nombre_modelo='net1', generar_tabla=True, results_dir='resultados'):
    """
    Visualiza los pesos del modelo NET1 (regresión logística) como un mapa de calor
    y genera una tabla con los pesos más significativos.

    Args:
        model: Modelo NET1 entrenado
        nombre_modelo: Nombre para guardar la visualización
        generar_tabla: Si True, genera y devuelve una tabla con las métricas
        results_dir: Directorio donde guardar los resultados

    Returns:
        dict: Métricas de los pesos si generar_tabla es True
    """
    vis_results_dir = configure_output_dir(results_dir, 'pesos')

    # Obtenemos los pesos y el sesgo
    pesos = model.fc.weight.data.cpu().numpy()
    bias = model.fc.bias.data.cpu().numpy()

    # El modelo NET1 tiene pesos de forma [1, input_size]
    # Reorganizamos a la forma de la imagen original [IMG_SIZE, IMG_SIZE]
    from utils.data_utils import IMG_SIZE
    pesos_imagen = pesos.reshape(1, IMG_SIZE, IMG_SIZE)

    # Visualizamos
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 1, 1)
    img = plt.imshow(pesos_imagen[0], cmap='viridis')
    plt.colorbar(img, label='Valor del peso')
    plt.title(f'Pesos de {nombre_modelo} (sesgo: {bias[0]:.4f})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, f'pesos_{nombre_modelo}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Generamos estadísticas generales
    metricas = extraer_metricas_pesos(pesos, nombre_modelo)
    metricas['bias'] = float(bias[0])

    # Identificamos los pesos más influyentes
    pesos_flat = pesos.flatten()
    top_positivos_idx = np.argsort(pesos_flat)[-10:]  # Top 10 pesos positivos
    top_negativos_idx = np.argsort(pesos_flat)[:10]  # Top 10 pesos negativos

    # Creamos listas para almacenar información sobre pesos influyentes
    top_pesos = []

    # Procesamos los pesos positivos (favorecen Lepto)
    for idx in top_positivos_idx[::-1]:  # Invertimos para mostrar de mayor a menor
        row = idx // IMG_SIZE
        col = idx % IMG_SIZE
        valor = pesos_flat[idx]
        top_pesos.append({
            'fila': int(row),
            'columna': int(col),
            'valor': float(valor),
            'favorece': 'Lepto',
            'magnitud_normalizada': float(valor / metricas['max'])
        })

    # Procesamos los pesos negativos (favorecen Kunzea)
    for idx in top_negativos_idx:
        row = idx // IMG_SIZE
        col = idx % IMG_SIZE
        valor = pesos_flat[idx]
        top_pesos.append({
            'fila': int(row),
            'columna': int(col),
            'valor': float(valor),
            'favorece': 'Kunzea',
            'magnitud_normalizada': float(abs(valor) / abs(metricas['min']))
        })

    # Añadimos la lista de pesos más influyentes a las métricas
    metricas['top_pesos'] = top_pesos

    # Mostramos un resumen textual
    print(f"Análisis de pesos de {nombre_modelo}:")
    print(f"Rango de valores: [{metricas['min']:.4f}, {metricas['max']:.4f}]")
    print(f"Media: {metricas['media']:.4f}, Desviación estándar: {metricas['desv_std']:.4f}")
    print(f"Porcentaje de pesos cercanos a cero: {metricas['pct_cerca_cero']:.2f}%")
    print(f"Porcentaje de pesos positivos: {metricas['pct_positivos']:.2f}%")
    print(f"Porcentaje de pesos negativos: {metricas['pct_negativos']:.2f}%")

    if generar_tabla:
        # Creamos un DataFrame con las métricas generales
        metricas_df = pd.DataFrame([{
            'Modelo': nombre_modelo,
            'Media': metricas['media'],
            'Desv_Std': metricas['desv_std'],
            'Min': metricas['min'],
            'Max': metricas['max'],
            'Rango': metricas['rango'],
            'Pct_Cerca_Cero': metricas['pct_cerca_cero'],
            'Pct_Positivos': metricas['pct_positivos'],
            'Pct_Negativos': metricas['pct_negativos'],
            'Num_Pesos': metricas['num_pesos'],
            'Bias': metricas['bias']
        }])

        # Guardamos las métricas generales
        metricas_csv_path = os.path.join(vis_results_dir, f'metricas_{nombre_modelo}.csv')
        metricas_df.to_csv(metricas_csv_path, index=False)
        print(f"Métricas generales guardadas en: {metricas_csv_path}")

        # Creamos un DataFrame para los pesos más influyentes
        pesos_data = []
        for peso in metricas['top_pesos']:
            peso_data = {
                'Modelo': nombre_modelo,
                'Fila': peso['fila'],
                'Columna': peso['columna'],
                'Valor': peso['valor'],
                'Favorece': peso['favorece'],
                'Magnitud_Normalizada': peso['magnitud_normalizada']
            }
            pesos_data.append(peso_data)

        pesos_df = pd.DataFrame(pesos_data)
        pesos_csv_path = os.path.join(vis_results_dir, f'top_pesos_{nombre_modelo}.csv')
        pesos_df.to_csv(pesos_csv_path, index=False)
        print(f"Top pesos influyentes guardados en: {pesos_csv_path}")

        return metricas
    return None


def visualizar_pesos_net2(model, nombre_modelo='net2', num_neuronas=9, generar_tabla=True, results_dir='resultados'):
    """
    Visualiza los pesos de algunas neuronas de la primera capa oculta de NET2.

    Args:
        model: Modelo NET2 entrenado
        nombre_modelo: Nombre para guardar la visualización
        num_neuronas: Número de neuronas a visualizar
        generar_tabla: Si True, genera y devuelve una tabla con las métricas
        results_dir: Directorio donde guardar los resultados

    Returns:
        dict: Métricas de los pesos si generar_tabla es True
    """
    vis_results_dir = configure_output_dir(results_dir, 'pesos')

    # Obtenemos los pesos de la primera capa
    # En NET2, el modelo es una secuencia que empieza con flatten y sigue con capas lineales
    primera_capa = None
    for module in model.model:
        if isinstance(module, nn.Linear):
            primera_capa = module
            break

    if primera_capa is None:
        print("No se pudo encontrar la primera capa lineal en el modelo NET2")
        return None

    # Obtenemos los pesos
    pesos = primera_capa.weight.data.cpu().numpy()

    # Determinamos cuántas neuronas visualizar (mínimo entre el parámetro y el número real)
    num_neuronas = min(num_neuronas, pesos.shape[0])

    # Reorganizamos los pesos de cada neurona a la forma de la imagen
    from utils.data_utils import IMG_SIZE
    pesos_imagenes = pesos[:num_neuronas].reshape(num_neuronas, IMG_SIZE, IMG_SIZE)

    # Determinamos filas y columnas para la visualización
    n_filas = int(np.ceil(np.sqrt(num_neuronas)))
    n_cols = int(np.ceil(num_neuronas / n_filas))

    # Visualizamos
    plt.figure(figsize=(15, 15))

    for index in range(num_neuronas):
        plt.subplot(n_filas, n_cols, index + 1)
        img = plt.imshow(pesos_imagenes[index], cmap='viridis')
        plt.colorbar(img)
        plt.title(f'Neurona {index + 1}')
        plt.axis('off')

    plt.suptitle(f'Pesos de la primera capa oculta de {nombre_modelo}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_results_dir, f'pesos_{nombre_modelo}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Generamos estadísticas para todas las neuronas
    neuronas_metricas = []

    for index in range(pesos.shape[0]):
        # Calculamos métricas para cada neurona
        neurona_pesos = pesos[index].reshape(IMG_SIZE, IMG_SIZE)
        metricas = extraer_metricas_pesos(neurona_pesos, f"{nombre_modelo}_neurona_{index + 1}")

        # Identificamos los patrones principales que detecta esta neurona
        pesos_flat = neurona_pesos.flatten()
        max_pos_idx = np.argmax(pesos_flat)
        max_neg_idx = np.argmin(pesos_flat)

        max_pos_row, max_pos_col = max_pos_idx // IMG_SIZE, max_pos_idx % IMG_SIZE
        max_neg_row, max_neg_col = max_neg_idx // IMG_SIZE, max_neg_idx % IMG_SIZE

        metricas['max_pos_ubicacion'] = (int(max_pos_row), int(max_pos_col))
        metricas['max_neg_ubicacion'] = (int(max_neg_row), int(max_neg_col))

        # Determinamos si la neurona parece detectar bordes, texturas, etc.
        gradiente_h = np.mean(np.abs(np.diff(neurona_pesos, axis=1)))
        gradiente_v = np.mean(np.abs(np.diff(neurona_pesos, axis=0)))

        metricas['gradiente_h'] = float(gradiente_h)
        metricas['gradiente_v'] = float(gradiente_v)

        # Estimamos qué tipo de detector parece ser
        if gradiente_h > 2 * metricas['desv_std'] or gradiente_v > 2 * metricas['desv_std']:
            if gradiente_h > gradiente_v:
                metricas['tipo_detector'] = 'borde_horizontal'
            else:
                metricas['tipo_detector'] = 'borde_vertical'
        elif gradiente_h > 1.5 * metricas['desv_std'] and gradiente_v > 1.5 * metricas['desv_std']:
            metricas['tipo_detector'] = 'esquina'
        elif metricas['pct_cerca_cero'] > 80:
            metricas['tipo_detector'] = 'localizado'
        else:
            metricas['tipo_detector'] = 'textura'

        neuronas_metricas.append(metricas)

    # Calculamos métricas globales para todo el conjunto de pesos
    metricas_globales = extraer_metricas_pesos(pesos, nombre_modelo)

    # Mostramos un resumen textual
    print(f"Análisis global de pesos de {nombre_modelo}:")
    print(f"Rango de valores: [{metricas_globales['min']:.4f}, {metricas_globales['max']:.4f}]")
    print(f"Media: {metricas_globales['media']:.4f}, Desv. estándar: {metricas_globales['desv_std']:.4f}")
    print(f"Porcentaje de pesos cercanos a cero: {metricas_globales['pct_cerca_cero']:.2f}%")

    # Contamos los diferentes tipos de detectores
    tipos_detectores = {}
    for m in neuronas_metricas:
        tipo = m['tipo_detector']
        tipos_detectores[tipo] = tipos_detectores.get(tipo, 0) + 1

    print("\nTipos de detectores identificados:")
    for tipo, cantidad in tipos_detectores.items():
        print(f"  {tipo}: {cantidad} neuronas ({cantidad / len(neuronas_metricas) * 100:.1f}%)")

    if generar_tabla:
        # Preparamos un DataFrame con las métricas globales
        global_df = pd.DataFrame([{
            'Modelo': nombre_modelo,
            'Media': metricas_globales['media'],
            'Desv_Std': metricas_globales['desv_std'],
            'Pct_Cerca_Cero': metricas_globales['pct_cerca_cero'],
            'Pct_Positivos': metricas_globales['pct_positivos'],
            'Pct_Negativos': metricas_globales['pct_negativos'],
            'Rango': metricas_globales['rango'],
            'Num_Pesos': metricas_globales['num_pesos']
        }])

        # Guardamos las métricas globales
        global_csv_path = os.path.join(vis_results_dir, f'pesos_global_{nombre_modelo}.csv')
        global_df.to_csv(global_csv_path, index=False)
        print(f"Métricas globales guardadas en: {global_csv_path}")

        # Preparamos un DataFrame para las métricas de cada neurona
        neuronas_data = []
        for idx, neurona in enumerate(neuronas_metricas):
            neurona_data = {
                'Modelo': nombre_modelo,
                'Neurona': idx + 1,
                'Media': neurona['media'],
                'Desv_Std': neurona['desv_std'],
                'Pct_Cerca_Cero': neurona['pct_cerca_cero'],
                'Tipo_Detector': neurona['tipo_detector'],
                'Gradiente_H': neurona['gradiente_h'],
                'Gradiente_V': neurona['gradiente_v'],
                'Max_Pos_Ubicacion': f"{neurona['max_pos_ubicacion']}",
                'Max_Neg_Ubicacion': f"{neurona['max_neg_ubicacion']}"
            }
            neuronas_data.append(neurona_data)

        # Creamos el DataFrame de neuronas y lo guardamos
        neuronas_df = pd.DataFrame(neuronas_data)
        neuronas_csv_path = os.path.join(vis_results_dir, f'pesos_neuronas_{nombre_modelo}.csv')
        neuronas_df.to_csv(neuronas_csv_path, index=False)
        print(f"Métricas por neurona guardadas en: {neuronas_csv_path}")

        # Preparamos un DataFrame para los tipos de detectores
        detectores_data = [{'Tipo': tipo, 'Cantidad': cantidad, 'Porcentaje': cantidad / len(neuronas_metricas) * 100}
                           for tipo, cantidad in tipos_detectores.items()]
        detectores_df = pd.DataFrame(detectores_data)
        detectores_csv_path = os.path.join(vis_results_dir, f'tipos_detectores_{nombre_modelo}.csv')
        detectores_df.to_csv(detectores_csv_path, index=False)
        print(f"Distribución de tipos de detectores guardada en: {detectores_csv_path}")

        resultados = {
            'global': metricas_globales,
            'neuronas': neuronas_metricas,
            'tipos_detectores': tipos_detectores
        }
        return resultados
    return None


def analizar_pesos_cnn(model, nombre_modelo='cnn_base', generar_tabla=True, results_dir='resultados'):
    """
    Analiza cuantitativamente los pesos de todas las capas convolucionales de un modelo CNN.

    Args:
        model: Modelo CNN entrenado
        nombre_modelo: Nombre del modelo para guardar resultados
        generar_tabla: Si True, genera y devuelve una tabla con las métricas
        results_dir: Directorio donde guardar los resultados (usado para configurar el directorio de salida)

    Returns:
        dict: Métricas de los pesos convolucionales si generar_tabla es True
    """

    # Obtenemos todas las capas convolucionales
    capas_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    resultados = {
        'modelo': nombre_modelo,
        'num_capas_conv': len(capas_conv),
        'capas': []
    }

    print(f"\nAnálisis de pesos convolucionales para {nombre_modelo}:")
    print(f"Número de capas convolucionales: {len(capas_conv)}")

    # Analizamos cada capa
    for index, capa in enumerate(capas_conv):
        info_capa = {
            'capa_idx': index,
            'input_channels': capa.in_channels,
            'output_channels': capa.out_channels,
            'kernel_size': capa.kernel_size[0],
            'num_parametros': capa.weight.numel()
        }

        # Obtenemos los pesos de la capa
        pesos = capa.weight.data.cpu().numpy()

        # Calculamos métricas estadísticas
        metricas = extraer_metricas_pesos(pesos, f"{nombre_modelo}_capa_{index}")
        info_capa.update(metricas)

        # Calculamos normas L2 de cada filtro (para identificar los más importantes)
        normas_filtros = np.zeros(capa.out_channels)
        for filtro_idx in range(capa.out_channels):
            normas_filtros[filtro_idx] = np.linalg.norm(pesos[filtro_idx])

        # Identificamos los filtros con mayor norma (más importantes)
        top_filtros_idx = np.argsort(normas_filtros)[-5:][::-1]  # Top 5, de mayor a menor

        info_capa['top_filtros'] = [{
            'filtro_idx': int(idx),
            'norma': float(normas_filtros[idx]),
            'norma_normalizada': float(normas_filtros[idx] / np.max(normas_filtros))
        } for idx in top_filtros_idx]

        resultados['capas'].append(info_capa)

        # Mostramos un resumen para esta capa
        print(f"\nCapa {index + 1}:")
        print(f"  Canales de entrada: {capa.in_channels}, Canales de salida: {capa.out_channels}")
        print(f"  Tamaño del kernel: {capa.kernel_size[0]}x{capa.kernel_size[0]}")
        print(f"  Rango de valores: [{metricas['min']:.4f}, {metricas['max']:.4f}]")
        print(f"  Media: {metricas['media']:.4f}, Desv. estándar: {metricas['desv_std']:.4f}")
        print(f"  Filtros más importantes (por norma L2):", end=" ")
        print(", ".join([f"Filtro {idx}" for idx in top_filtros_idx]))

    if generar_tabla:
        # Creamos un DataFrame para guardar los resultados en un formato tabular
        datos_tabla = []
        for capa in resultados['capas']:
            # Extraemos los datos principales de cada capa
            fila = {
                'Modelo': nombre_modelo,
                'Capa': f"Capa {capa['capa_idx'] + 1}",
                'Media': capa['media'],
                'Desv_Std': capa['desv_std'],
                'Rango': capa['rango'],
                'Top_Filtros': ", ".join([str(tf['filtro_idx']) for tf in capa['top_filtros'][:3]])
            }
            datos_tabla.append(fila)

        # Convertimos a DataFrame
        df = pd.DataFrame(datos_tabla)

        # Configuramos el directorio de salida para análisis de pesos
        vis_results_dir = configure_output_dir(results_dir, 'pesos')

        # Guardamos como CSV
        csv_path = os.path.join(vis_results_dir, f'pesos_cnn_{nombre_modelo}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResultados guardados en: {csv_path}")

        return resultados
    return None


def generar_tabla_comparativa_pesos(metricas_modelos, tipo='basicos', results_dir='resultados'):
    """
    Genera una tabla comparativa de las métricas de pesos de varios modelos.
    Muestra los resultados por consola y los guarda en formato CSV.

    Args:
        metricas_modelos: Lista de diccionarios con métricas de cada modelo
        tipo: Tipo de modelos ('basicos' para NET1/NET2, 'cnn' para modelos CNN)
        results_dir: Directorio donde guardar los resultados

    Returns:
        None
    """
    # Usamos subdirectorio 'tablas' para organizar mejor
    vis_results_dir = configure_output_dir(results_dir, 'pesos')

    if tipo == 'basicos':
        # Columnas para modelos básicos
        columnas = ['Modelo', 'Media', 'Desv. Std.', '% Cerca Cero', '% Positivos', '% Negativos', 'Rango', 'Sesgo']

        # Preparamos los datos
        datos_tabla = []
        for metricas in metricas_modelos:
            fila = [
                metricas['modelo'],
                f"{metricas['media']:.4f}",
                f"{metricas['desv_std']:.4f}",
                f"{metricas['pct_cerca_cero']:.2f}%",
                f"{metricas['pct_positivos']:.2f}%",
                f"{metricas['pct_negativos']:.2f}%",
                f"{metricas['rango']:.4f}",
                f"{metricas.get('bias', 'N/A')}"
            ]
            datos_tabla.append(fila)

        titulo = "Comparativa de pesos de modelos básicos"
        nombre_archivo = "tabla_pesos_basicos.csv"

    elif tipo == 'cnn':
        # Columnas para modelos CNN
        columnas = ['Modelo', 'Capa', 'Media', 'Desv. Std.', 'Rango', 'Top Filtros']

        # Preparamos los datos
        datos_tabla = []
        for modelo in metricas_modelos:
            for capa in modelo['capas']:
                fila = [
                    modelo['modelo'],
                    f"Capa {capa['capa_idx'] + 1}",
                    f"{capa['media']:.4f}",
                    f"{capa['desv_std']:.4f}",
                    f"{capa['rango']:.4f}",
                    ", ".join([str(tf['filtro_idx']) for tf in capa['top_filtros'][:3]])
                ]
                datos_tabla.append(fila)

        titulo = "Comparativa de pesos de modelos CNN"
        nombre_archivo = "tabla_pesos_cnn.csv"

    elif tipo == 'activaciones':
        # Columnas para activaciones por clase
        columnas = ['Modelo', 'Capa', 'Act. Kunzea', 'Act. Lepto', 'Dif. Media', 'Filtros K/L', 'Top Discriminativos']

        # Preparamos los datos
        datos_tabla = []
        for modelo in metricas_modelos:
            fila = [
                modelo['modelo'],
                f"Capa {modelo['capa_idx'] + 1}",
                f"{modelo['kunzea_activacion_media']:.4f}",
                f"{modelo['lepto_activacion_media']:.4f}",
                f"{modelo['diferencia_media']:.4f}",
                modelo['ratio_filtros'],
                ", ".join([str(idx) for idx in modelo['top_filtros_discriminativos'][:3]])
            ]
            datos_tabla.append(fila)

        titulo = "Comparativa de activaciones por clase"
        nombre_archivo = "tabla_activaciones_clase.csv"

    else:
        print(f"Tipo de tabla no reconocido: {tipo}")
        return

    # Convertimos los datos a un DataFrame
    df = pd.DataFrame(datos_tabla, columns=columnas)

    # Guardamos el CSV
    csv_path = os.path.join(vis_results_dir, nombre_archivo)
    df.to_csv(csv_path, index=False)
    print(f"Tabla guardada en formato CSV: {csv_path}")

    # Mostramos la tabla en formato texto por consola
    print(f"\n=== {titulo} ===")
    # Imprimimos encabezados
    cabecera = " | ".join(columnas)
    print(cabecera)
    print("-" * len(cabecera))
    # Imprimimos filas
    for fila in datos_tabla:
        print(" | ".join(str(valor) for valor in fila))