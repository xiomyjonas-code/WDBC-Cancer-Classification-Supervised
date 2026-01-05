# Supervised Machine Learning: Breast Cancer Diagnosis (WDBC) 🧬 
**Author:** xiomyjonas-code   
**Area:** Bioinformática (Algoritmos e Inteligencia Artificial)

Este proyecto presenta un análisis comparativo utilizando algoritmos de **Aprendizaje Automático Supervisado** para la clasificación de tumores de mama (Malignos vs. Benignos) utilizando el conjunto de datos *Wisconsin Diagnostic Breast Cancer (WDBC)*.

El estudio contrasta el rendimiento de modelos modernos (SVM, Random Forest, k-NN) frente a los estándares de referencia establecidos en la literatura médica clásica (Wolberg et al., 1995).

## Objetivos 🎯 
1. **Optimización de Modelos:** Implementar y ajustar hiperparámetros para Support Vector Machines (SVM), Random Forest (RF) y k-Nearest Neighbors (k-NN).
2. **Validación Cruzada:** Garantizar la robustez de los resultados mediante *10-fold Cross-Validation*.
3. **Benchmarking:** Comparar la precisión diagnóstica del "Gold Standard" (3 variables nucleares) frente a modelos entrenados con selección de características optimizada (21 variables).

## Stack Tecnológico 🛠️ 
* **Lenguaje:** R
* **Machine Learning:** `caret` (Training & Tuning), `kernlab` (SVM), `randomForest`, `class`.
* **Visualización & Análisis:** `pROC` (Curvas ROC), `ggplot2`, `corrplot`.

## Metodología y Afinación de Modelos 🔬 

Se realizó una búsqueda de hiperparámetros (*Tuning*) para maximizar el Área Bajo la Curva (AUC).

### 1. Random Forest
Se evaluó el número de predictores seleccionados aleatoriamente en cada división (*mtry*).
* **Resultado Óptimo:** El modelo alcanzó su máximo rendimiento (ROC $\approx$ 0.9895) con **15 predictores** (`mtry = 15`). Aumentar a 20 predictores resultó en una ligera caída del rendimiento por sobreajuste.

### 2. k-Nearest Neighbors (k-NN)
Se analizó el impacto del tamaño del vecindario (*k*) en la capacidad de generalización.
* **Resultado Óptimo:** Se identificó un pico de rendimiento (ROC $\approx$ 0.985) con **$k = 21$ vecinos**. Esto sugiere que una frontera de decisión más suave favorece la clasificación en este dataset.

### 3. Support Vector Machine (SVM)
Se implementó un SVM con **Kernel Lineal**. Debido a la clara separabilidad lineal de los datos en el hiperespacio, el modelo convergió óptimamente con el parámetro de coste por defecto, logrando un desempeño superior sin necesidad de kernels radiales complejos.

## Resultados Finales 📊

El modelo **SVM Lineal** demostró ser el clasificador más robusto, superando ligeramente a Random Forest y k-NN en las métricas finales sobre el conjunto de prueba.

| Modelo | AUC (ROC) | Sensibilidad (Maligno)* | Especificidad (Benigno)* |
| :--- | :---: | :---: | :---: |
| **SVM Lineal** | **0.998** | **92.8%** | **98.6%** |
| Random Forest | 0.997 | 92.8% | 98.5% |
| k-NN ($k=21$) | 0.985 | 92.8% | 97.5% |

*> Nota: Métricas ajustadas considerando "Maligno" como la clase positiva de interés clínico.*

## 💡 Conclusión
El análisis valida que, aunque los modelos no lineales como Random Forest y k-NN ofrecen un rendimiento excelente ($>97\%$ AUC), la naturaleza de los datos biológicos del WDBC permite una clasificación casi perfecta utilizando un separador lineal (SVM). Esto respalda el principio de parsimonia en bioinformática clínica: modelos más simples y explicables pueden ser igual o más efectivos que arquitecturas complejas.

## 📂 Estructura del Repositorio
* `supervisado_analysis.R`: Código fuente completo en R.
* `data/`: carpeta con los archivos `data.csv`(Dataset WDBC) y `variables.csv` (información del dataset)
* `plots/`: Carpeta con gráficos de afinación y curvas ROC.
---

*Proyecto desarrollado para el Máster en Bioinformática.*
