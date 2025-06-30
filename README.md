# semantic-building-segmentation

Este repositorio contiene el código y los resultados del Trabajo de Fin de Grado (TFG) enfocado en la segmentación semántica de edificios a partir de imágenes aéreas RGB, utilizando modelos de aprendizaje profundo.

## Objetivo del Proyecto

Desarrollar y comparar diferentes arquitecturas de segmentación semántica para la detección automática de edificios, optimizando tareas como la planificación urbana, el análisis cartográfico y la gestión territorial.

## Estructura del Repositorio

semantic-building-segmentation/
│
├── data-preprocessing/ # Scripts para preparar y transformar los datos (PNOA, INRIA)
│ ├── Transformando.ipynb
│ ├── LimpiezaPNOA.ipynb
| ├── 512x512.ipynb
│ └── cortar_imagenes.py
│
├── models/ # Entrenamiento de cada modelo y configuración
|
├── results/ # Resultados obtenidos en diferentes experimentos
│ ├── metrics/ # Métricas .npy y .json por escenario
│ ├── plots/ # Gráficas de entrenamiento
│ └── predictions/ # Ejemplos visuales
│
└── README.md

## Modelos Evaluados

- **UNet-ResNet34**
- **DeepLabV3+ - ResNeXt101 (32x8d)**
- **PAN - MobileNetV2**
- **UPerNet - ConvNeXt Base**

Los modelos fueron entrenados y evaluados en tres escenarios:

1. INRIA (preentrenamiento)
2. Fine-tuning con datos del PNOA
3. Entrenamiento conjunto con INRIA + PNOA

## Datos Utilizados

- **INRIA Aerial Image Labeling Dataset**
- **PNOA (Plan Nacional de Ortofotografía Aérea)**: imágenes RGB y máscaras corregidas manualmente. Si está interesado en las imágenes utilizadas no dude en contactar a jorge.pastor@alumnos.upm.es
- En la fase final, se combinan ambos datasets para mejorar la generalización.

## Resultados

Se analizaron métricas como IoU, F1-score y Accuracy, tanto en validación como en pruebas visuales sobre zonas reales de España. Puedes encontrar los resultados en:

results/metrics/
results/plots/
results/predictions/

## Requisitos

- Python 3.9+
- PyTorch y PyTorch Lightning
- segmentation_models_pytorch
- Albumentations
- OpenCV, matplotlib, numpy, pandas
- QGIS (para evaluación geográfica final)

## Ejecución

1. Preprocesa los datos con los scripts de `data-preprocessing/`.
2. Entrena los modelos con los notebooks o scripts dentro de `models/`.
3. Visualiza los resultados.

## Notas Finales

Este trabajo destaca la importancia de los datos geográficamente representativos y la validación manual para tareas de segmentación en el mundo real. Se exploran retos, resultados y propuestas de mejora para futuras aplicaciones.

---

