# ************************************************************************
# * analysis/__init__.py
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
Módulo de análisis para la clasificación de granos de polen.
Este paquete contiene funciones para el análisis de modelos, visualización
y evaluación del rendimiento.
"""

from .data_loader import (
    load_training_histories,
    load_test_metrics,
    load_models,
    load_data,
    obtener_imagen_muestra
)

from .visualization import (
    plot_training_history,
    grafica_comparativa_exactitud,
    plot_learning_curves,
    plot_optimized_vs_final_comparison
)

from .weights_analysis import (
    visualizar_pesos_conv,
    visualizar_pesos_net1,
    visualizar_pesos_net2,
    analizar_pesos_cnn,
    extraer_metricas_pesos,
    generar_tabla_comparativa_pesos
)

from .feature_maps import (
    visualizar_mapas_caracteristicas,
    analizar_mapas_caracteristicas_cuantitativo,
    comparar_activaciones_por_clase_cuantitativo,
    analizar_filtros_discriminativos,
    comparar_activaciones_por_clase
)

from .performance import (
    generar_tabla_comparativa,
    analizar_activaciones_modelos_basicos,
    analizar_matrices_confusion,
    analizar_metricas_por_clase,
    generar_tabla_resumen_modelos,
    comparar_optimizados_vs_finales,
    generar_tabla_comparativa_opt_final,
)

from .data_augmentation import (
    analisis_cualitativo_completo,
    analizar_errores_por_clase
)
