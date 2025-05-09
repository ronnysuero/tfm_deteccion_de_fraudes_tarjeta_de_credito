# Detección de Fraudes en Tarjetas de Crédito con Aprendizaje Automático

Este repositorio contiene el Jupyter Notebook y los recursos asociados al proyecto de investigación "Detección de Fraudes en Tarjetas de Crédito en el Sector Bancario". El objetivo principal es implementar, comparar y analizar la efectividad de diferentes modelos de aprendizaje automático (Árbol de Decisión, SVM, Red Neuronal) y estrategias de manejo de desbalanceo de clases (SMOTE y GANs).

## Descripción del Proyecto

El fraude en transacciones con tarjetas de crédito representa un desafío significativo para las instituciones financieras. Este proyecto explora la aplicación de varios algoritmos de Machine Learning para identificar transacciones fraudulentas, utilizando un [dataset público de Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ampliamente conocido. Se comparan los modelos mencionados y se evalúa el impacto de SMOTE versus un enfoque basado en Redes Generativas Adversarias (GANs) para el aumento de datos de la clase minoritaria.

El informe completo en LaTeX, que detalla la revisión de literatura, metodología, resultados y conclusiones [`Ver Informe Completo (PDF)`](https://drive.google.com/file/d/10zC0tlyUsgdLIiF9yqsG9t3p-Gxpa5__/view).

## Contenido del Repositorio

* `creditcard.ipynb`: El Jupyter Notebook principal que contiene todo el código Python para:
  * Carga y preprocesamiento del dataset.
  * Análisis Exploratorio de Datos (EDA), incluyendo manejo de duplicados y visualizaciones.
  * Implementación de modelos: Árbol de Decisión (DT), Máquina de Soporte Vectorial (SVM), Red Neuronal (MLP).
  * Manejo de desbalanceo de clases:
    * SMOTE (Synthetic Minority Over-sampling Technique).
    * Implementación de una GAN (Generative Adversarial Network) para generar datos sintéticos de fraude (definición de Generador y Discriminador, bucle de entrenamiento, generación de muestras y escalado inverso).
  * Entrenamiento de clasificadores con datos balanceados por SMOTE y datos aumentados por GAN.
  * Evaluación de los modelos utilizando métricas como Matriz de Confusión, Precision, Recall, F1-Score, AUC ROC y AUC PR.
  * Generación de visualizaciones (histogramas, box plots, matrices de confusión, curvas de pérdida de la GAN).
* `creditcard.csv`: Contiene todas las transacciones realizadas con tarjetas de crédito europeas en septiembre de 2013.
* `readme.md`: Este archivo.

## Dataset Utilizado

El proyecto utiliza el dataset "Credit Card Fraud Detection" disponible en Kaggle:
* **Enlace:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Descripción:** Contiene transacciones realizadas con tarjetas de crédito europeas en septiembre de 2013. Incluye 30 características (Tiempo, Monto y 28 características anonimizadas V1-V28 resultado de PCA) y una variable objetivo 'Class' (0 para legítima, 1 para fraude). Es un dataset altamente desbalanceado.

## Requisitos e Instalación

El notebook fue desarrollado en Python (versión 3.11.7) y requiere las siguientes librerías principales:

* pandas (versión 2.1.4)
* numpy (versión 1.26.4)
* scikit-learn (versión 1.2.2)
* imbalanced-learn (versión 0.11.0) - Para SMOTE
* tensorflow (versión 2.16.0 o compatible) - Para Keras y la implementación de GAN/NN
* matplotlib (versión 3.8.0)
* seaborn (versión 0.12.2)
* jupyterlab o jupyter notebook

Puedes instalar las dependencias usando pip. Se recomienda crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows

pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn jupyterlab