# Detección de Neumonía a partir de Imágenes de Rayos X

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para detectar neumonía a partir de imágenes de rayos X. El proyecto incluye la limpieza de datos, el entrenamiento del modelo, la evaluación del rendimiento y la creación de una API REST para realizar predicciones.

## Estructura del Proyecto

- `data/`: Contiene los datos en diferentes estados (crudos, procesados, entrenamiento, prueba, validación).
- `notebooks/`: Contiene los Jupyter Notebooks utilizados para el análisis y desarrollo.
- `src/`: Código fuente del proyecto.
- `models/`: Modelos entrenados.
- `reports/`: Informes y presentaciones.

## Cómo Ejecutar el Proyecto

1. Crear y activar el entorno virtual (opcional):
    ```bash
    python -m venv env
    source env/bin/activate  # Unix/Mac
    env\Scripts\activate  # Windows
    ```

2. Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecutar el preprocesamiento de datos:
    ```bash
    python src/data_preprocessing.py
    ```

4. Entrenar el modelo:
    ```bash
    python src/model_training.py
    ```

5. Evaluar el modelo:
    ```bash
    python src/model_evaluation.py
    ```

6. Ejecutar la API:
    ```bash
    python src/api.py
    ```

## Autor
Luis Contreras Alcalde

## Licencia
MIT License
