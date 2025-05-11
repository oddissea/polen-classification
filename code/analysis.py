# ************************************************************************
# * analysis.py
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
import analysis
import os
from datetime import datetime

if __name__ == "__main__":
    # 1. Configuración inicial
    results_dir = 'resultados'
    print(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Cargar datos
    print("Cargando datos procesados...")
    train_loader_aug, train_loader_normal, val_loader, test_loader = analysis.load_data()
    print(f"Total de imágenes de entrenamiento: {len(train_loader_aug.dataset)}")
    print(f"Total de imágenes de validación: {len(val_loader.dataset)}")
    print(f"Total de imágenes de test: {len(test_loader.dataset)}")

    # Creas un diccionario con los dataloaders
    dataloaders = {
        'train_aug': train_loader_aug,
        'train_normal': train_loader_normal,
        'val': val_loader,
        'test': test_loader
    }

    # 3. Cargar historiales y métricas
    print("\n--- Cargando historiales y métricas ---")
    historiales = analysis.load_training_histories(results_dir)
    test_metrics = analysis.load_test_metrics(results_dir)

    # 4. Cargar modelos
    print("\n--- Cargando modelos ---")
    # Cargamos ambas versiones: los mejores modelos (con early stopping) y los modelos finales
    print("Cargando modelos optimizados (best)...")
    modelos_best = analysis.load_models(results_dir, load_best=True)
    print("Cargando modelos finales...")
    modelos_final = analysis.load_models(results_dir, load_best=False)

    # Para la mayoría de los análisis (pesos, mapas de características, etc.),
    # usamos los modelos finales por defecto
    modelos = modelos_final

    # 5. Obtener imágenes de muestra
    print("\n--- Obteniendo imágenes de muestra ---")
    img_kunzea, img_kunzea_np = analysis.obtener_imagen_muestra(val_loader, clase=0)
    img_lepto, img_lepto_np = analysis.obtener_imagen_muestra(val_loader, clase=1)

    # 6. ANÁLISIS EXHAUSTIVO DE ENTRENAMIENTO
    print("\n--- Generando curvas de validación comparativas ---")
    # Definimos todos los modelos a comparar
    modelos_a_comparar = [
        'NET1_normal', 'NET1_aug',
        'NET2_normal', 'NET2_aug',
        'CNN_base_normal', 'CNN_base_aug',
        'CNN_variant1_normal', 'CNN_variant1_aug',
        'CNN_variant2_normal', 'CNN_variant2_aug'
    ]

    # 6.1 Curvas de validación para todos los modelos (similar a Fig. 11.11 de Hastie)
    analysis.plot_learning_curves(
        historiales,
        modelos_a_comparar,
        mode="validation",
        title="Comparativa general de modelos durante el aprendizaje",
        filename="curvas_validation_comparativas.png",
        show_early_stopping=True,  # Mostramos los puntos de early stopping
        split_graphs=False,
        split_by_aug=True,
        smoothing_factor=0.7,  # Nivel moderado de suavizado (ajusta según necesidad)
        save_both_versions=True,  # Guarda tanto la original como la suavizada
        results_dir=results_dir
    )

    # 6.1.1 Curvas de test para todos los modelos
    analysis.plot_learning_curves(
        historiales,
        modelos_a_comparar,
        mode="test",
        test_metrics=test_metrics,
        title="Rendimiento en test por modelo",
        filename="curvas_test_comparativas.png",
        show_early_stopping=True,  # Mostramos los puntos de early stopping
        split_graphs=False,
        split_by_aug=True,
        smoothing_factor=0.7,  # Nivel moderado de suavizado (ajusta según necesidad)
        save_both_versions=True,  # Guarda tanto la original como la suavizada
        results_dir=results_dir
    )

    # 6.2 Comparativas de modelos básicos # visualizaciones/exactitud
    subfolder = "/exactitud/"
    print("\n--- Visualizaciones comparativas de modelos ---")
    print("Generando gráficas para modelos básicos...")

    # Comparativa de modelos básicos con augmentation
    analysis.plot_training_history(
        [historiales['NET1_aug'], historiales['NET2_aug']],
        ["NET1 (aug)", "NET2 (aug)"],
        "Comparativa de modelos básicos con data augmentation",
        "curvas_modelos_basicos_aug.png",
        results_dir
    )

    # Comparativa de modelos básicos sin augmentation
    analysis.plot_training_history(
        [historiales['NET1_normal'], historiales['NET2_normal']],
        ["NET1 (normal)", "NET2 (normal)"],
        "Comparativa de modelos básicos sin data augmentation",
        "curvas_modelos_basicos_normal.png",
        results_dir
    )

    # Efecto del data augmentation en NET1 y NET2
    analysis.plot_training_history(
        [historiales['NET1_normal'], historiales['NET1_aug']],
        ["NET1 (normal)", "NET1 (aug)"],
        "Efecto del data augmentation en NET1",
        "efecto_aug_net1.png",
        results_dir
    )

    analysis.plot_training_history(
        [historiales['NET2_normal'], historiales['NET2_aug']],
        ["NET2 (normal)", "NET2 (aug)"],
        "Efecto del data augmentation en NET2",
        "efecto_aug_net2.png",
        results_dir
    )

    # 6.3 Comparativas de modelos CNN
    print("Generando gráficas para modelos CNN...")

    # Comparativa de modelos CNN con augmentation
    analysis.plot_training_history(
        [historiales['CNN_base_aug'], historiales['CNN_variant1_aug'], historiales['CNN_variant2_aug']],
        ["CNN Base (aug)", "CNN Variante 1 (aug)", "CNN Variante 2 (aug)"],
        "Comparativa de modelos CNN con data augmentation",
        "curvas_modelos_cnn_aug.png",
        results_dir
    )

    # Comparativa de modelos CNN sin augmentation
    analysis.plot_training_history(
        [historiales['CNN_base_normal'], historiales['CNN_variant1_normal'], historiales['CNN_variant2_normal']],
        ["CNN Base (normal)", "CNN Variante 1 (normal)", "CNN Variante 2 (normal)"],
        "Comparativa de modelos CNN sin data augmentation",
        "curvas_modelos_cnn_normal.png",
        results_dir
    )

    # Efecto de data augmentation en cada modelo CNN
    for modelo_base in ['CNN_base', 'CNN_variant1', 'CNN_variant2']:
        analysis.plot_training_history(
            [historiales[f'{modelo_base}_normal'], historiales[f'{modelo_base}_aug']],
            [f"{modelo_base} (normal)", f"{modelo_base} (aug)"],
            f"Efecto del data augmentation en {modelo_base}",
            f"efecto_aug_{modelo_base.lower()}.png",
            results_dir
        )

    # 6.4 Comparativa general entre todos los modelos
    print("Generando comparativas generales...")

    # Con augmentation
    analysis.plot_training_history(
        [
            historiales['NET1_aug'],
            historiales['NET2_aug'],
            historiales['CNN_base_aug'],
            historiales['CNN_variant1_aug'],
            historiales['CNN_variant2_aug']
        ],
        [
            "NET1 (aug)",
            "NET2 (aug)",
            "CNN Base (aug)",
            "CNN Variante 1 (aug)",
            "CNN Variante 2 (aug)"
        ],
        "Comparativa general de modelos con data augmentation",
        "comparativa_general_aug.png",
        results_dir
    )

    # Sin augmentation
    analysis.plot_training_history(
        [
            historiales['NET1_normal'],
            historiales['NET2_normal'],
            historiales['CNN_base_normal'],
            historiales['CNN_variant1_normal'],
            historiales['CNN_variant2_normal']
        ],
        [
            "NET1 (normal)",
            "NET2 (normal)",
            "CNN Base (normal)",
            "CNN Variante 1 (normal)",
            "CNN Variante 2 (normal)"
        ],
        "Comparativa general de modelos sin data augmentation",
        "comparativa_general_normal.png",
        results_dir
    )

    # 6.5 Tabla comparativa de resultados
    print("\n--- Generando tabla comparativa de resultados ---")
    analysis.generar_tabla_comparativa(historiales, modelos_a_comparar, results_dir)

    # 6.6 Gráfica comparativa con métricas de test
    print("\n--- Generando gráfica de exactitud comparativa con resultados de test ---")
    analysis.grafica_comparativa_exactitud(historiales, modelos_a_comparar, test_metrics, results_dir)

    # 7. ANÁLISIS EXHAUSTIVO DE PESOS
    print("\n--- Analizando pesos de modelos ---")
    metricas_net1 = {}
    metricas_net2 = {}
    metricas_cnn = {}

    # 7.1 Análisis de pesos NET1
    print("\n--- Analizando pesos de modelos NET1 ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'NET1' in n]:
        print(f"Analizando {nombre}...")
        metricas_net1[nombre] = analysis.visualizar_pesos_net1(model, nombre, results_dir=results_dir)

    # 7.2 Análisis de pesos NET2
    print("\n--- Analizando pesos de modelos NET2 ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'NET2' in n]:
        print(f"Analizando {nombre}...")
        metricas_net2[nombre] = analysis.visualizar_pesos_net2(model, nombre, results_dir=results_dir)

    # 7.3 Análisis de pesos CNN
    print("\n--- Analizando pesos de modelos CNN ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'CNN' in n]:
        print(f"Analizando {nombre}...")
        metricas_cnn[nombre] = analysis.analizar_pesos_cnn(model, nombre, results_dir=results_dir)

        # Añadir visualización de pesos para cada capa convolucional
        num_capas = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        for capa_idx in range(min(3, num_capas)):  # Visualizamos las primeras 3 capas como máximo
            analysis.visualizar_pesos_conv(
                model,
                capa_idx=capa_idx,
                nombre_modelo=nombre,
                num_filtros=16,  # Mostramos 16 filtros como máximo
                results_dir=results_dir
            )
            print(f"  Visualizados pesos de la capa {capa_idx + 1} para {nombre}")

    # 7.4 Análisis de activaciones de modelos básicos
    print("\n--- Analizando activaciones de modelos básicos ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'NET' in n and 'CNN' not in n]:
        analysis.analizar_activaciones_modelos_basicos(model, test_loader, modelo_nombre=nombre,
                                                       results_dir=results_dir)

    # 8. ANÁLISIS EXHAUSTIVO DE MAPAS DE CARACTERÍSTICAS
    print("\n--- Análisis exhaustivo de mapas de características ---")

    # 8.1 Visualización de mapas de características para todos los modelos CNN y todas las capas
    print("\n--- Visualizando mapas de características ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'CNN' in n]:
        num_capas = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        for capa_idx in range(num_capas):
            # Para Kunzea
            analysis.visualizar_mapas_caracteristicas(
                model, img_kunzea,
                capa_idx=capa_idx,
                nombre_modelo=f"{nombre}_kunzea",
                results_dir=results_dir
            )

            # Para Lepto
            # if capa_idx == 0: # Para primera capa solamente para no generar demasiados gráficos
            analysis.visualizar_mapas_caracteristicas(
                model, img_lepto,
                capa_idx=capa_idx,
                nombre_modelo=f"{nombre}_lepto",
                results_dir=results_dir
            )

    # 8.2 Análisis de diferencias entre clases para cada capa de cada modelo CNN
    print("\n--- Análisis de diferencias entre clases ---")
    activaciones_por_clase = []

    for nombre, model in [(n, m) for n, m in modelos.items() if 'CNN' in n]:
        print(f"Comparando activaciones para {nombre}...")
        num_capas = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        for capa_idx in range(num_capas):
            resultados, top_filtros = analysis.comparar_activaciones_por_clase_cuantitativo(
                model, img_kunzea, img_lepto,
                capa_idx=capa_idx,
                nombre_modelo=nombre,
                results_dir=results_dir
            )
            if resultados:
                activaciones_por_clase.append(resultados)

            if top_filtros is not None:
                print(f"{nombre} - Filtros más discriminativos (capa {capa_idx + 1}): {top_filtros}")

    # 8.3 Análisis de filtros discriminativos para cada modelo CNN
    print("\n--- Análisis de filtros discriminativos ---")
    for nombre, model in [(n, m) for n, m in modelos.items() if 'CNN' in n]:
        print(f"Analizando filtros discriminativos para {nombre}...")
        num_capas = min(3, len([m for m in model.modules() if isinstance(m, nn.Conv2d)]))
        for capa_idx in range(num_capas):
            analysis.analizar_filtros_discriminativos(
                model, img_kunzea, img_lepto,
                capa_idx=capa_idx,
                nombre_modelo=nombre,
                results_dir=results_dir
            )

    # 9. ANÁLISIS CUALITATIVO COMPLETO DEL DATA AUGMENTATION
    print("\n--- Iniciando análisis cualitativo completo ---")

    # 9.1 Análisis de matrices de confusión
    models_base = ['NET1', 'NET2', 'CNN_base', 'CNN_variant1', 'CNN_variant2']
    analysis.analizar_matrices_confusion(test_metrics, models_base, results_dir)

    # 9.2 Análisis de métricas por clase
    analysis.analizar_metricas_por_clase(test_metrics, models_base, results_dir)

    # 9.3 Análisis de errores específicos
    modelos_para_analisis_errores = ['CNN_base_normal', 'CNN_base_aug', 'CNN_variant1_normal', 'CNN_variant1_aug']
    dataloaders = {'test': test_loader}
    analysis.analizar_errores_por_clase(modelos, dataloaders, test_metrics, modelos_para_analisis_errores, results_dir)

    # 9.4 Análisis cualitativo completo del impacto del data augmentation
    analysis.analisis_cualitativo_completo(test_metrics, modelos, dataloaders, results_dir)

    # 10. TABLAS COMPARATIVAS Y RESÚMENES
    print("\n--- Generando tablas comparativas y resúmenes ---")

    # 10.1 Tabla comparativa de pesos
    analysis.generar_tabla_comparativa_pesos(list(metricas_net1.values()), tipo='basicos', results_dir=results_dir)
    analysis.generar_tabla_comparativa_pesos([m['global'] for m in metricas_net2.values()], tipo='basicos',
                                             results_dir=results_dir)
    analysis.generar_tabla_comparativa_pesos(list(metricas_cnn.values()), tipo='cnn', results_dir=results_dir)
    analysis.generar_tabla_comparativa_pesos(activaciones_por_clase, tipo='activaciones', results_dir=results_dir)

    # 10.2 Tabla resumen de todos los modelos
    analysis.generar_tabla_resumen_modelos(metricas_net1, metricas_net2, metricas_cnn, test_metrics, results_dir)

    # 10.3 Resumen de resultados en test
    print("\n--- Resumen de resultados de test ---")
    print(f"{'Modelo':<20} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)

    for modelo in modelos_a_comparar:
        if modelo in test_metrics:
            metrics = test_metrics[modelo]
            print(
                f"{modelo:<20} {metrics['test_acc']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}"
            )

    # 10.4 Análisis de la mejora debido al data augmentation
    print("\n--- Mejora debido al data augmentation (en test) ---")
    for modelo_base in models_base:
        normal_key = f"{modelo_base}_normal"
        aug_key = f"{modelo_base}_aug"

        if normal_key in test_metrics and aug_key in test_metrics:
            mejora = (test_metrics[aug_key]['test_acc'] - test_metrics[normal_key]['test_acc']) * 100
            print(f"{modelo_base}: {mejora:.2f}%")

    # 10.5 ANÁLISIS COMPARATIVO: MODELOS OPTIMIZADOS VS FINALES
    print("\n=== ANÁLISIS COMPARATIVO: MODELOS OPTIMIZADOS VS FINALES ===")

    # Cargamos las métricas de test para modelos finales
    print("Cargando métricas de test de modelos finales...")
    test_metrics_final = analysis.load_test_metrics(results_dir, load_final=True)

    # Tabla comparativa
    print("\n--- Generando tabla comparativa entre modelos optimizados y finales ---")
    analysis.generar_tabla_comparativa_opt_final(test_metrics, test_metrics_final, models_base, results_dir)

    # Matrices de confusión
    print("\n--- Analizando matrices de confusión: optimizados vs finales ---")
    analysis.analizar_matrices_confusion(test_metrics, models_base, results_dir, test_metrics_final)

    # Visualizaciones detalladas
    print("\n--- Generando visualizaciones comparativas ---")
    for model_base in models_base:
        model_keys = [f"{model_base}_normal", f"{model_base}_aug"]
        analysis.plot_optimized_vs_final_comparison(
            historiales,
            test_metrics,
            test_metrics_final,
            model_keys,
            title=f"Impacto del entrenamiento completo: {model_base}",
            filename=f"optimized_vs_final_{model_base}.png",
            results_dir=results_dir
        )

    # Análisis global
    print("\n--- Comparando rendimiento global entre modelos optimizados y finales ---")
    analysis.comparar_optimizados_vs_finales(
        test_metrics,
        test_metrics_final,
        models_base,
        historiales,
        results_dir
    )


    # 11. RESUMEN FINAL
    print("\n=== Análisis completado ===")
    print(f"Visualizaciones guardadas en: {os.path.join(results_dir, 'visualizaciones')}")

    # 11.1 Imprimimos los tiempos de entrenamiento (si están disponibles)
    print("\n--- Tiempos de entrenamiento ---")
    print(f"{'Modelo':<20} {'Tiempo (s)':<10} {'Épocas':<10}")
    print("-" * 40)

    for modelo in modelos_a_comparar:
        if modelo in historiales and 'training_time' in historiales[modelo]:
            tiempo = historiales[modelo]['training_time']
            epocas = historiales[modelo].get('stopped_epoch', len(historiales[modelo]['train_acc']))
            print(f"{modelo:<20} {tiempo:.2f}       {epocas}")

    print(f"Fecha y hora de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")