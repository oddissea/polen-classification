# ************************************************************************
# * analysis/feature_maps.py
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
Módulo para el análisis de mapas de características y activaciones de los modelos CNN.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from .visualization import configure_output_dir


def preparar_imagen_tensor(imagen, device):
    """
    Prepara una imagen para procesamiento, convirtiéndola a tensor si es necesario.

    Args:
        imagen: Imagen como array numpy o tensor
        device: Dispositivo donde cargar la imagen

    Returns:
        torch.Tensor: Imagen preparada como tensor
    """
    # Preparamos la imagen
    if isinstance(imagen, np.ndarray):
        if len(imagen.shape) == 2:  # Si es una imagen en escala de grises
            imagen = imagen[np.newaxis, np.newaxis, :, :]
        elif len(imagen.shape) == 3:  # Si es RGB
            imagen = imagen.transpose(2, 0, 1)[np.newaxis, :, :, :]
        imagen = torch.FloatTensor(imagen)

    # Movemos al dispositivo adecuado
    return imagen.to(device)


def visualizar_mapas_caracteristicas(model, imagen, capa_idx=0, nombre_modelo='cnn', num_mapas=16,
                                     results_dir='resultados'):
    """
    Visualiza los mapas de características generados en una capa convolucional para una imagen.

    Args:
        model: Modelo CNN entrenado
        imagen: Tensor o array de la imagen de entrada
        capa_idx: Índice de la capa convolucional
        nombre_modelo: Nombre para guardar la visualización
        num_mapas: Número máximo de mapas a visualizar
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'mapas')
    device = next(model.parameters()).device
    model.eval()

    # Preparamos la imagen usando la función auxiliar
    imagen = preparar_imagen_tensor(imagen, device)

    # Lista para almacenar las activaciones
    activaciones = []

    # Función hook para capturar activaciones
    def hook_fn(_, __, output):
        # Parámetros no utilizados: module, input_val
        activaciones.append(output.detach())

    # Encontrar la capa convolucional específica
    capas_conv = []

    # Primero extraemos todas las capas convolucionales
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            capas_conv.append(module)

    # Verificamos que tenemos suficientes capas
    if capa_idx >= len(capas_conv):
        print(f"Error: El model solo tiene {len(capas_conv)} capas convolucionales")
        return

    # Registramos un hook para la capa específica
    hook = capas_conv[capa_idx].register_forward_hook(hook_fn)

    # Pasamos la imagen por el model
    with torch.no_grad():
        _ = model(imagen)

    # Eliminamos el hook
    hook.remove()

    # Verificamos que tenemos activaciones
    if not activaciones:
        print(f"Error: No se capturaron activaciones para la capa {capa_idx}")
        return

    # Obtenemos los mapas de características de la capa especificada
    feature_maps = activaciones[0].cpu().numpy().squeeze()

    # Si los feature maps son de un solo canal (la entrada era un batch de 1)
    if len(feature_maps.shape) == 2:
        feature_maps = feature_maps[np.newaxis, :, :]

    # Limitamos el número de mapas a visualizar
    num_mapas = min(num_mapas, feature_maps.shape[0])

    # Determinamos el número de filas y columnas para la visualización
    n_filas = int(np.ceil(np.sqrt(num_mapas)))
    n_cols = int(np.ceil(num_mapas / n_filas))

    # Creamos la figura
    plt.figure(figsize=(12, 12))

    # Visualizamos cada mapa de características
    for index in range(num_mapas):
        plt.subplot(n_filas, n_cols, index + 1)

        img = feature_maps[index]
        # Normalizamos para mejor visualización
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        plt.imshow(img, cmap='viridis')
        plt.title(f'Mapa {index + 1}')
        plt.axis('off')

    plt.suptitle(f'Mapas de características - Capa conv {capa_idx + 1} - {nombre_modelo}')
    plt.tight_layout()

    # Guardamos la visualización
    plt.savefig(os.path.join(vis_results_dir, f'mapas_caract_conv{capa_idx + 1}_{nombre_modelo}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def analizar_mapas_caracteristicas_cuantitativo(model, imagen, capa_idx=0, nombre_modelo='cnn', generar_tabla=True,
                                                results_dir='resultados'):
    """
    Analiza cuantitativamente los mapas de características generados por una capa convolucional.

    Args:
        model: Modelo CNN entrenado
        imagen: Tensor o array de una imagen de entrada
        capa_idx: Índice de la capa convolucional a analizar
        nombre_modelo: Nombre del modelo para guardar resultados
        generar_tabla: Si True, genera y devuelve una tabla con las métricas
        results_dir: Directorio donde guardar los resultados (no usado directamente en esta función)

    Returns:
        dict: Métricas de los mapas de características si generar_tabla es True
    """
    # Configura el directorio de salida para posibles visualizaciones futuras
    vis_results_dir = configure_output_dir(results_dir, 'mapas')

    device = next(model.parameters()).device
    model.eval()

    # Preparamos la imagen usando la función auxiliar
    imagen = preparar_imagen_tensor(imagen, device)

    # Lista para almacenar las activaciones
    activaciones = []

    # Función hook para capturar activaciones
    def hook_fn(_, __, output):
        # Parámetros no utilizados: module, input_val
        activaciones.append(output.detach())

    # Encontrar la capa convolucional específica
    capas_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    # Verificamos que tenemos suficientes capas
    if capa_idx >= len(capas_conv):
        print(f"Error: El modelo solo tiene {len(capas_conv)} capas convolucionales")
        return None

    # Registramos un hook para la capa específica
    hook = capas_conv[capa_idx].register_forward_hook(hook_fn)

    # Pasamos la imagen por el modelo
    with torch.no_grad():
        _ = model(imagen)

    # Eliminamos el hook
    hook.remove()

    # Verificamos que tenemos activaciones
    if not activaciones:
        print(f"Error: No se capturaron activaciones para la capa {capa_idx}")
        return None

    # Obtenemos los mapas de características de la capa especificada
    feature_maps = activaciones[0].cpu().numpy().squeeze()

    # Si los feature maps son de un solo canal (la entrada era un batch de 1)
    if len(feature_maps.shape) == 2:
        feature_maps = feature_maps[np.newaxis, :, :]

    num_canales = feature_maps.shape[0]

    # Preparamos un diccionario para almacenar los resultados
    resultados = {
        'modelo': nombre_modelo,
        'capa_idx': capa_idx,
        'num_canales': num_canales,
        'dim_espacial': feature_maps.shape[1:],
        'canales': []
    }

    # Analizamos cada canal (mapa de características)
    for i in range(num_canales):
        mapa = feature_maps[i]

        umbral = 0.1 * np.max(mapa)

        metricas = {
            'canal_idx': i,
            'media': float(np.mean(mapa)),
            'mediana': float(np.median(mapa)),
            'desv_std': float(np.std(mapa)),
            'min': float(np.min(mapa)),
            'max': float(np.max(mapa)),
            'rango': float(np.max(mapa) - np.min(mapa)),
            'activacion_media': float(np.mean(mapa)),
            'pct_activado': float(np.mean(mapa > 0) * 100),
            'sparsity': float(np.mean(mapa < umbral) * 100)
        }

        # Encontramos la ubicación de la máxima activación
        max_idx = np.unravel_index(np.argmax(mapa), mapa.shape)
        metricas['max_activacion_pos'] = (int(max_idx[0]), int(max_idx[1]))

        # Calculamos el centro de masa de la activación
        # (indica dónde se concentra la activación)
        indices = np.indices(mapa.shape)
        activacion_normalizada = mapa - np.min(mapa)
        suma_activacion = np.sum(activacion_normalizada)

        if suma_activacion > 0:
            centro_y = np.sum(indices[0] * activacion_normalizada) / suma_activacion
            centro_x = np.sum(indices[1] * activacion_normalizada) / suma_activacion
            metricas['centro_activacion'] = (float(centro_y), float(centro_x))
        else:
            metricas['centro_activacion'] = (float('nan'), float('nan'))

        resultados['canales'].append(metricas)

    # Calculamos estadísticas globales
    activacion_media_global = np.mean([c['activacion_media'] for c in resultados['canales']])
    sparsity_media = np.mean([c['sparsity'] for c in resultados['canales']])

    resultados['activacion_media_global'] = float(activacion_media_global)
    resultados['sparsity_media'] = float(sparsity_media)

    # Identificamos los canales más activados
    canales_por_activacion = sorted(resultados['canales'],
                                    key=lambda x: x['activacion_media'],
                                    reverse=True)
    top_canales = [c['canal_idx'] for c in canales_por_activacion[:5]]

    resultados['top_canales_activacion'] = top_canales

    # Mostramos resumen
    print(f"\nAnálisis cuantitativo de mapas de características - {nombre_modelo}, capa {capa_idx}:")
    print(f"Número de canales: {num_canales}")
    print(f"Dimensión espacial: {resultados['dim_espacial']}")
    print(f"Activación media global: {activacion_media_global:.4f}")
    print(f"Sparsity media: {sparsity_media:.2f}%")
    print(f"Canales más activados: {top_canales}")

    if generar_tabla:
        # Generamos un CSV con las estadísticas globales
        global_stats = {
            'Modelo': [nombre_modelo],
            'Capa': [capa_idx],
            'Num_Canales': [num_canales],
            'Dimension_Espacial': [str(resultados['dim_espacial'])],
            'Activacion_Media_Global': [activacion_media_global],
            'Sparsity_Media': [sparsity_media],
            'Top_Canales': [str(top_canales)]
        }

        global_df = pd.DataFrame(global_stats)
        global_csv_path = os.path.join(vis_results_dir, f'mapas_stats_global_{nombre_modelo}_capa{capa_idx}.csv')
        global_df.to_csv(global_csv_path, index=False)
        print(f"Estadísticas globales guardadas en: {global_csv_path}")

        # Creamos un DataFrame con las métricas detalladas de cada canal
        canales_data = []
        for canal in resultados['canales']:
            canal_data = {
                'Modelo': nombre_modelo,
                'Capa': capa_idx,
                'Canal_Idx': canal['canal_idx'],
                'Media': canal['media'],
                'Mediana': canal['mediana'],
                'Desv_Std': canal['desv_std'],
                'Min': canal['min'],
                'Max': canal['max'],
                'Rango': canal['rango'],
                'Activacion_Media': canal['activacion_media'],
                'Pct_Activado': canal['pct_activado'],
                'Sparsity': canal['sparsity'],
                'Max_Activacion_Pos': str(canal['max_activacion_pos']),
                'Centro_Activacion': str(canal['centro_activacion'])
            }
            canales_data.append(canal_data)

        canales_df = pd.DataFrame(canales_data)
        canales_csv_path = os.path.join(vis_results_dir, f'mapas_stats_canales_{nombre_modelo}_capa{capa_idx}.csv')
        canales_df.to_csv(canales_csv_path, index=False)
        print(f"Estadísticas por canal guardadas en: {canales_csv_path}")

        return resultados
    return None


def obtener_activaciones_por_clase(model, img_clase0, img_clase1, capa_idx=0, nombre_modelo='cnn'):
    """
    Obtiene las activaciones medias por canal para dos clases diferentes.

    Args:
        model: Modelo CNN
        img_clase0: Tensor de imagen de la clase 0 (Kunzea)
        img_clase1: Tensor de imagen de la clase 1 (Lepto)
        capa_idx: Índice de la capa convolucional
        nombre_modelo: Nombre del modelo para mensajes de error

    Returns:
        Tuple: (act_clase0_mean, act_clase1_mean, diferencia) o (None, None, None) si hay error
    """
    device = next(model.parameters()).device
    model.eval()

    def obtener_activaciones(imagen):
        activaciones = []

        def hook_fn(_, __, output):
            # Parámetros no utilizados: module_data, input_data
            activaciones.append(output.detach())

        # Encontramos la capa convolucional específica
        capas_conv = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                capas_conv.append(module)

        if capa_idx >= len(capas_conv):
            return None

        hook = capas_conv[capa_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(imagen.to(device))

        hook.remove()

        if not activaciones:
            return None

        return activaciones[0].cpu().numpy().squeeze()

    # Preparamos las imágenes usando la función auxiliar
    img_clase0 = preparar_imagen_tensor(img_clase0, device) if not torch.is_tensor(
        img_clase0) or img_clase0.device != device else img_clase0
    img_clase1 = preparar_imagen_tensor(img_clase1, device) if not torch.is_tensor(
        img_clase1) or img_clase1.device != device else img_clase1

    # Obtenemos las activaciones para ambas clases
    act_clase0 = obtener_activaciones(img_clase0)
    act_clase1 = obtener_activaciones(img_clase1)

    if act_clase0 is None or act_clase1 is None:
        print(f"No se pudieron obtener activaciones para el modelo {nombre_modelo}")
        return None, None, None

    # Calculamos la media por canal para cada clase
    if act_clase0.ndim > 2:
        act_clase0_mean = np.mean(act_clase0, axis=(1, 2))
        act_clase1_mean = np.mean(act_clase1, axis=(1, 2))
    else:
        act_clase0_mean = np.mean(act_clase0, axis=0)
        act_clase1_mean = np.mean(act_clase1, axis=0)

    # Calculamos la diferencia absoluta entre las activaciones
    diferencia = np.abs(act_clase0_mean - act_clase1_mean)

    return act_clase0_mean, act_clase1_mean, diferencia


def comparar_activaciones_por_clase(model, img_clase0, img_clase1, capa_idx=0, nombre_modelo='cnn',
                                    results_dir='resultados'):
    """
    Compara las activaciones medias de una capa para dos clases diferentes.

    Args:
        model: Modelo CNN
        img_clase0: Tensor de imagen de la clase 0 (Kunzea)
        img_clase1: Tensor de imagen de la clase 1 (Lepto)
        capa_idx: Índice de la capa convolucional
        nombre_modelo: Nombre del modelo para guardar la visualización
        results_dir: Directorio donde guardar los resultados

    Returns:
        Tuple: (top_filters, diferencia)
    """
    vis_results_dir = configure_output_dir(results_dir, 'activaciones')

    # Utilizamos la función común para obtener activaciones
    act_clase0_mean, act_clase1_mean, diferencia = obtener_activaciones_por_clase(
        model, img_clase0, img_clase1, capa_idx, nombre_modelo
    )

    if act_clase0_mean is None:
        return None, None

    # Visualizamos los resultados
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    indices = np.arange(len(act_clase0_mean))
    plt.bar(indices, act_clase0_mean, alpha=0.5, label='Kunzea (clase 0)')
    plt.bar(indices, act_clase1_mean, alpha=0.5, label='Lepto (clase 1)')
    plt.xlabel('Índice del filtro')
    plt.ylabel('Activación media')
    plt.title(f'Activaciones medias por filtro - Capa {capa_idx + 1}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(indices, diferencia)
    plt.xlabel('Índice del filtro')
    plt.ylabel('|Diferencia de activación|')
    plt.title('Diferencia absoluta de activación entre clases')

    # Destacamos los 5 filtros con mayor diferencia
    top_filters = np.argsort(diferencia)[-5:]
    for index in top_filters:
        plt.annotate(f"{index}", xy=(index, diferencia[index]), xytext=(0, 5),
                     textcoords="offset points", ha='center', color='red')

    plt.suptitle(f'Comparación de activaciones por clase - {nombre_modelo} - Capa {capa_idx + 1}')
    plt.tight_layout()

    plt.savefig(os.path.join(vis_results_dir, f'comp_activaciones_{nombre_modelo}_capa{capa_idx + 1}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return top_filters, diferencia


def comparar_activaciones_por_clase_cuantitativo(model, img_clase0, img_clase1, capa_idx=0, nombre_modelo='cnn',
                                                 generar_tabla=True, results_dir='resultados'):
    """
    Compara cuantitativamente las activaciones de una capa para dos clases diferentes.

    Args:
        model: Modelo CNN
        img_clase0: Tensor de imagen de la clase 0 (Kunzea)
        img_clase1: Tensor de imagen de la clase 1 (Lepto)
        capa_idx: Índice de la capa convolucional
        nombre_modelo: Nombre del modelo para guardar resultados
        generar_tabla: Si True, genera y devuelve una tabla con las métricas
        results_dir: Directorio donde guardar los resultados (no usado directamente en esta función)

    Returns:
        dict: Métricas comparativas si generar_tabla es True
    """
    # Configura el directorio de salida para posibles visualizaciones futuras
    vis_results_dir = configure_output_dir(results_dir, 'activaciones')

    # Utilizamos la función común para obtener activaciones
    act_clase0_mean, act_clase1_mean, diferencia = obtener_activaciones_por_clase(
        model, img_clase0, img_clase1, capa_idx, nombre_modelo
    )

    if act_clase0_mean is None:
        return None, None

    # Preparamos un diccionario para almacenar los resultados
    resultados = {
        'modelo': nombre_modelo,
        'capa_idx': capa_idx,
        'num_filtros': len(diferencia),
        'kunzea_activacion_media': float(np.mean(act_clase0_mean)),
        'lepto_activacion_media': float(np.mean(act_clase1_mean)),
        'diferencia_media': float(np.mean(diferencia)),
        'filtros': []
    }

    # Analizamos cada filtro
    for i in range(len(diferencia)):
        info_filtro = {
            'filtro_idx': i,
            'kunzea_activacion': float(act_clase0_mean[i]),
            'lepto_activacion': float(act_clase1_mean[i]),
            'diferencia_abs': float(diferencia[i]),
            'ratio_activacion': float(act_clase1_mean[i] / (act_clase0_mean[i] + 1e-10)),  # Evitamos división por cero
            'favorece': 'Lepto' if act_clase1_mean[i] > act_clase0_mean[i] else 'Kunzea'
        }

        resultados['filtros'].append(info_filtro)

    # Ordenamos los filtros por diferencia de activación
    filtros_ordenados = sorted(resultados['filtros'],
                               key=lambda x: x['diferencia_abs'],
                               reverse=True)

    # Identificamos los filtros más discriminativos
    top_filtros = [f['filtro_idx'] for f in filtros_ordenados[:5]]
    resultados['top_filtros_discriminativos'] = top_filtros

    # Calculamos qué clase tiene mayor activación global
    if resultados['kunzea_activacion_media'] > resultados['lepto_activacion_media']:
        resultados['clase_mayor_activacion'] = 'Kunzea'
        resultados['ratio_activacion_clases'] = float(resultados['kunzea_activacion_media'] /
                                                      (resultados['lepto_activacion_media'] + 1e-10))
    else:
        resultados['clase_mayor_activacion'] = 'Lepto'
        resultados['ratio_activacion_clases'] = float(resultados['lepto_activacion_media'] /
                                                      (resultados['kunzea_activacion_media'] + 1e-10))

    # Contamos cuántos filtros favorecen a cada clase
    filtros_kunzea = sum(1 for f in resultados['filtros'] if f['favorece'] == 'Kunzea')
    filtros_lepto = sum(1 for f in resultados['filtros'] if f['favorece'] == 'Lepto')

    resultados['num_filtros_kunzea'] = filtros_kunzea
    resultados['num_filtros_lepto'] = filtros_lepto
    resultados['ratio_filtros'] = f"{filtros_kunzea}:{filtros_lepto}"

    # Mostramos un resumen
    print(f"\nComparación de activaciones por clase - {nombre_modelo}, capa {capa_idx}:")
    print(f"Número total de filtros: {len(diferencia)}")
    print(f"Activación media para Kunzea: {resultados['kunzea_activacion_media']:.4f}")
    print(f"Activación media para Lepto: {resultados['lepto_activacion_media']:.4f}")
    print(f"Diferencia media de activación: {resultados['diferencia_media']:.4f}")
    print(f"Filtros más discriminativos: {top_filtros}")
    print(f"Proporción de filtros por clase: {resultados['ratio_filtros']} (Kunzea:Lepto)")

    if generar_tabla:
        # Creamos un DataFrame con las estadísticas generales
        general_stats = {
            'Modelo': [nombre_modelo],
            'Capa': [capa_idx],
            'Num_Filtros': [len(diferencia)],
            'Kunzea_Activacion_Media': [resultados['kunzea_activacion_media']],
            'Lepto_Activacion_Media': [resultados['lepto_activacion_media']],
            'Diferencia_Media': [resultados['diferencia_media']],
            'Clase_Mayor_Activacion': [resultados['clase_mayor_activacion']],
            'Ratio_Activacion_Clases': [resultados['ratio_activacion_clases']],
            'Num_Filtros_Kunzea': [filtros_kunzea],
            'Num_Filtros_Lepto': [filtros_lepto],
            'Ratio_Filtros': [resultados['ratio_filtros']],
            'Top_Filtros_Discriminativos': [str(top_filtros)]
        }

        general_df = pd.DataFrame(general_stats)
        general_csv_path = os.path.join(vis_results_dir, f'act_clase_general_{nombre_modelo}_capa{capa_idx}.csv')
        general_df.to_csv(general_csv_path, index=False)
        print(f"Estadísticas generales de activación por clase guardadas en: {general_csv_path}")

        # Creamos un DataFrame con la información detallada de cada filtro
        filtros_data = []
        for filtro in resultados['filtros']:
            filtro_data = {
                'Modelo': nombre_modelo,
                'Capa': capa_idx,
                'Filtro_Idx': filtro['filtro_idx'],
                'Kunzea_Activacion': filtro['kunzea_activacion'],
                'Lepto_Activacion': filtro['lepto_activacion'],
                'Diferencia_Abs': filtro['diferencia_abs'],
                'Ratio_Activacion': filtro['ratio_activacion'],
                'Favorece': filtro['favorece'],
                'Es_Top_Discriminativo': filtro['filtro_idx'] in top_filtros
            }
            filtros_data.append(filtro_data)

        filtros_df = pd.DataFrame(filtros_data)
        filtros_csv_path = os.path.join(vis_results_dir, f'act_clase_filtros_{nombre_modelo}_capa{capa_idx}.csv')
        filtros_df.to_csv(filtros_csv_path, index=False)
        print(f"Información detallada de filtros guardada en: {filtros_csv_path}")

        return resultados, top_filtros

    return None, top_filtros


def analizar_filtros_discriminativos(model, img_kunzea_polen, img_lepto_polen, capa_idx=0, nombre_modelo='cnn',
                                     num_filtros=4, results_dir='resultados'):
    """
    Analiza los filtros más discriminativos comparando sus mapas de características
    para imágenes de diferentes clases.

    Args:
        model: Modelo CNN entrenado
        img_kunzea_polen: Tensor de imagen de Kunzea
        img_lepto_polen: Tensor de imagen de Lepto
        capa_idx: Índice de la capa convolucional
        nombre_modelo: Nombre del modelo para guardar visualización
        num_filtros: Número de filtros más discriminativos a analizar
        results_dir: Directorio donde guardar los resultados
    """
    vis_results_dir = configure_output_dir(results_dir, 'activaciones')
    device = next(model.parameters()).device

    # Obtenemos los filtros más discriminativos
    top_filtros, diferencias = comparar_activaciones_por_clase(model, img_kunzea_polen, img_lepto_polen, capa_idx,
                                                               nombre_modelo, results_dir)

    if top_filtros is None:
        return

    # Limitamos a los N filtros más discriminativos
    top_filtros = top_filtros[-num_filtros:]

    # Obtenemos los mapas de características para ambas clases
    activaciones_kunzea = []
    activaciones_lepto = []

    def hook_fn(activaciones):
        def _hook_fn(_, __, output):
            # Parámetros no utilizados: module_data, input_data
            activaciones.append(output.detach().cpu().numpy())

        return _hook_fn

    # Encontramos la capa convolucional específica
    capas_conv = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            capas_conv.append(module)

    # Verificamos que tenemos suficientes capas
    if capa_idx >= len(capas_conv):
        return

    # Preparamos las imágenes usando la función auxiliar
    img_kunzea_polen = preparar_imagen_tensor(img_kunzea_polen, device) if not torch.is_tensor(
        img_kunzea_polen) or img_kunzea_polen.device != device else img_kunzea_polen
    img_lepto_polen = preparar_imagen_tensor(img_lepto_polen, device) if not torch.is_tensor(
        img_lepto_polen) or img_lepto_polen.device != device else img_lepto_polen

    # Registramos hook para Kunzea
    hook_kunzea = capas_conv[capa_idx].register_forward_hook(hook_fn(activaciones_kunzea))
    # Pasamos la imagen
    with torch.no_grad():
        _ = model(img_kunzea_polen)
    # Eliminamos el hook
    hook_kunzea.remove()

    # Registramos hook para Lepto
    hook_lepto = capas_conv[capa_idx].register_forward_hook(hook_fn(activaciones_lepto))
    # Pasamos la imagen
    with torch.no_grad():
        _ = model(img_lepto_polen)
    # Eliminamos el hook
    hook_lepto.remove()

    if not activaciones_kunzea or not activaciones_lepto:
        return

    # Convertimos a matrices numpy y eliminamos dimensión de batch
    act_kunzea = activaciones_kunzea[0].squeeze()
    act_lepto = activaciones_lepto[0].squeeze()

    # Visualizamos los mapas de características para los filtros más discriminativos
    plt.figure(figsize=(15, 6))

    for index, filtro_idx in enumerate(top_filtros):
        # Kunzea
        plt.subplot(2, num_filtros, index + 1)
        mapa_kunzea = act_kunzea[filtro_idx]
        mapa_norm_kunzea = (mapa_kunzea - np.min(mapa_kunzea)) / (np.max(mapa_kunzea) - np.min(mapa_kunzea) + 1e-8)
        plt.imshow(mapa_norm_kunzea, cmap='viridis')
        plt.title(f'Filtro {filtro_idx} - Kunzea')
        plt.axis('off')

        # Lepto
        plt.subplot(2, num_filtros, index + 1 + num_filtros)
        mapa_lepto = act_lepto[filtro_idx]
        mapa_norm_lepto = (mapa_lepto - np.min(mapa_lepto)) / (np.max(mapa_lepto) - np.min(mapa_lepto) + 1e-8)
        plt.imshow(mapa_norm_lepto, cmap='viridis')
        plt.title(f'Filtro {filtro_idx} - Lepto')
        plt.axis('off')

    plt.suptitle(f'Filtros más discriminativos en capa {capa_idx + 1} - {nombre_modelo}')
    plt.tight_layout()

    # Guardamos la visualización
    plt.savefig(os.path.join(vis_results_dir, f'filtros_discrim_capa{capa_idx + 1}_{nombre_modelo}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()