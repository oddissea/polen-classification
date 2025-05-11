# Clasificación de Granos de Polen mediante Redes Neuronales Convolucionales

Este repositorio contiene la implementación y análisis de cinco arquitecturas neuronales para la clasificación automática de granos de polen morfológicamente similares. El proyecto forma parte de la asignatura de Minería de Datos del Máster en Investigación en Inteligencia Artificial (UNED).

## Resumen

Abordamos el problema de identificación automática de dos especies de polen neozelandesas consideradas indistinguibles por palinólogos. Desarrollamos y comparamos cinco modelos con complejidad creciente:

1. **NET1**: Red neuronal sin capa oculta (regresión logística binomial)
2. **NET2**: Red neuronal con capas ocultas (MLP)
3. **CNN Base**: Red neuronal convolucional estándar
4. **CNN Variante 1**: CNN con normalización por lotes y mayor número de filtros
5. **CNN Variante 2**: CNN con kernels heterogéneos (5×5, 3×3, 1×1)

Cada modelo se entrenó con y sin técnicas de aumento de datos conservadoras (rotaciones controladas, variaciones de brillo y desplazamientos), alcanzando una exactitud máxima de 96,94% en validación con la CNN Variante 2 con aumento de datos.

## Características principales

- **Pipeline completo** de entrenamiento y evaluación con PyTorch
- **Análisis exhaustivo** de pesos y mapas de características para interpretabilidad
- **Visualización** de filtros y activaciones por clase para comprender el aprendizaje
- **Estudio comparativo** del impacto del aumento de datos en diferentes arquitecturas
- **Análisis cualitativo** de la separabilidad de clases y detección de patrones morfológicos

## Estructura del repositorio

```
├── models/
│   ├── basic_models.py         # Implementación de NET1 y NET2
│   └── cnn_models.py           # Implementación de modelos CNN
├── utils/
│   └── data_utils.py           # Funciones para procesamiento de datos
├── analysis/
│   ├── visualization.py        # Herramientas de visualización 
│   ├── weights_analysis.py     # Análisis de pesos de modelos
│   ├── feature_maps.py         # Análisis de mapas de características
│   └── performance.py          # Evaluación de rendimiento
├── preprocess.py               # Preprocesamiento y aumento de datos
├── train.py                    # Entrenamiento de todos los modelos
├── analysis.py                 # Análisis exhaustivo de resultados
├── requirements.txt            # Dependencias del proyecto
└── memoria.pdf                 # Memoria detallada del proyecto
```

## Resultados principales

| Modelo | Data Aug. | Exactitud Val. | Exactitud Test |
|--------|-----------|----------------|----------------|
| NET1   | No        | 86.39%         | 84.44%         |
| NET1   | Sí        | 83.06%         | 79.17%         |
| NET2   | No        | 91.94%         | 91.94%         |
| NET2   | Sí        | 94.44%         | 93.06%         |
| CNN Base | No      | 95.56%         | 94.17%         |
| CNN Base | Sí      | 96.67%         | 94.72%         |
| CNN V1 | No        | 95.28%         | 95.83%         |
| CNN V1 | Sí        | 96.67%         | 95.83%         |
| CNN V2 | No        | 95.56%         | 94.17%         |
| CNN V2 | Sí        | 96.94%         | 93.89%         |

## Dependencias principales

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## Instalación

```bash
git clone https://github.com/oddissea/polen-classification.git
cd polen-classification
```

## Ejecución

Para ejecutar el preprocesamiento y aumento de datos:
```bash
python preprocess.py
```

Para entrenar todos los modelos:
```bash
python train.py
```

Para realizar el análisis completo:
```bash
python analysis.py
```

## Conclusiones

Nuestro análisis demuestra que las arquitecturas convolucionales superan ampliamente a los modelos básicos en esta tarea de clasificación. Sorprendentemente, el aumento de datos muestra un impacto variable según la arquitectura, no siempre traducido en mejoras de rendimiento. CNN Variante 1 sin aumento de datos ofrece el mejor equilibrio entre rendimiento (95,83% en test) y complejidad computacional, sugiriendo que un diseño arquitectónico apropiado puede ser más efectivo que la ampliación artificial del conjunto de datos para este problema específico.

## Autor

Fernando H. Nasser-Eddine López - Máster en Investigación en Inteligencia Artificial (UNED)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.