#### ---- 1. Preparación del Entorno y Librerías ---- ####

# Limpieza del entorno de trabajo
rm(list=ls())

# Nota: Se recomienda usar RProjects o rutas relativas para evitar rutas absolutas.
# Configuración del directorio de trabajo (Ajustar según usuario)
setwd("C:/Users/Tu directorio")

# Lista de paquetes requeridos
packs <- c("tidyverse", "caret", "corrplot", "pROC", "rpart", "kernlab")

# Verificación e instalación automática de paquetes faltantes
new_packs <- packs[!(packs %in% installed.packages()[,"Package"])]
if(length(new_packs)) install.packages(new_packs)

# Carga de librerías
lapply(packs, library, character.only = TRUE)


######################################################
#     INFORMACIÓN GENERAL Y METODOLÓGICA             #
######################################################
# El presente script analiza el dataset WDBC para la clasificación de tumores.
# Metodología: 
# 1. Selección de Características: Eliminación de variables correlacionadas.
# 2. Entrenamiento: Modelos SVM Lineal, Random Forest y KNN (10-fold CV).
# 3. Validación: Comparación entre modelo optimizado (21 vars) vs Paper (3 vars).
# 4. Métrica principal: Área Bajo la Curva (AUC).
#
# Semilla aleatoria de control: 1995
######################################################

#### ---- 2. Carga y Preprocesamiento de Datos ---- ####

# Carga del dataset
df <- read.csv("data.csv")

# 2.1. Limpieza inicial
# Eliminación de columna ID y columnas vacías
df$ID <- NULL 
df <- df[, colSums(is.na(df)) < nrow(df)]

# 2.2. Renombrado de variables (Estandarización)
nombres_base <- c("radius", "texture", "perimeter", "area", "smoothness",
                  "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")

nuevos_nombres <- c("Diagnosis", 
                    paste0(nombres_base, "_mean"),
                    paste0(nombres_base, "_se"),
                    paste0(nombres_base, "_worst"))

colnames(df) <- nuevos_nombres

# Conversión de la variable objetivo a Factor (M=Maligno, B=Benigno)
df$Diagnosis <- as.factor(df$Diagnosis)

#### ---- 3. Análisis de Correlación y Selección de Características ---- ####

# Selección de variables numéricas
datos_numericos <- df %>% select(-Diagnosis)

# Cálculo de matriz de correlación
cor_matrix <- cor(datos_numericos)

# Visualización gráfica de la correlación
par(mfrow=c(1,1))
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         order = "hclust", 
         tl.col = "black", 
         tl.cex = 0.6,
         title = "Matriz de Correlación WDBC",
         mar=c(0,0,2,0))

# 3.1. Tratamiento de Multicolinealidad
# Identificación de variables con correlación > 0.9 para su eliminación
highCorr <- findCorrelation(cor_matrix, cutoff = 0.9)
variables_a_eliminar <- colnames(datos_numericos)[highCorr]

cat("\n--- Selección de Características ---\n")
cat("Variables eliminadas por redundancia:", length(variables_a_eliminar), "\n")

# Creación del dataset optimizado (df_clean)
df_clean <- df %>% select(-all_of(variables_a_eliminar))


#### ---- 4. Configuración del Entrenamiento (Train/Test Split) ---- ####

# Definición de datasets para los dos escenarios
# Escenario 1: Dataset Optimizado (21 variables) -> df_clean
# Escenario 2: Dataset Paper (3 variables) -> df_paper
vars_paper <- c("Diagnosis", "area_worst", "smoothness_worst", "texture_mean")
df_paper <- df %>% dplyr::select(all_of(vars_paper))

# Partición de datos (Semilla 1995)
set.seed(1995)
trainIndex <- createDataPartition(df_clean$Diagnosis, p = 0.8, list = FALSE)

# Conjuntos de entrenamiento y prueba (Optimizado)
trainData_clean <- df_clean[trainIndex,]
testData_clean  <- df_clean[-trainIndex,]

# Conjuntos de entrenamiento y prueba (Paper)
trainData_paper <- df_paper[trainIndex,]
testData_paper  <- df_paper[-trainIndex,]

# Control de entrenamiento (Validación Cruzada 10 iteraciones)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)


#### ---- 5. Entrenamiento de Modelos ---- ####

cat("\n--- Iniciando Entrenamiento de Modelos ---\n")

# 5.1. SVM Lineal (Optimizado)
# Ideal para espacios multidimensionales linealmente separables.
print("Entrenando SVM Lineal (Dataset Optimizado)...")
svmClean <- train(Diagnosis ~., data = trainData_clean,
                  method = "svmLinear",
                  trControl = ctrl,
                  preProcess = c("center", "scale"),
                  metric = "ROC")

# 5.2. Random Forest (Optimizado)
print("Entrenando Random Forest (Dataset Optimizado)...")
rfClean <- train(Diagnosis ~., data = trainData_clean,
                 method = "rf",
                 trControl = ctrl,
                 preProcess = c("center", "scale"),
                 metric = "ROC",
                 tuneLength = 5)

# 5.3. k-NN (Optimizado)
print("Entrenando k-NN (Dataset Optimizado)...")
knnClean <- train(Diagnosis ~., data = trainData_clean,
                  method = "knn",
                  trControl = ctrl,
                  preProcess = c("center", "scale"),
                  tuneLength = 10,
                  metric = "ROC")

# 5.4. Modelo de Referencia (Paper - 3 Variables)
print("Entrenando Modelo Paper (3 variables)...")
modelPaper <- train(Diagnosis ~., data = trainData_paper,
                    method = "svmLinear",
                    trControl = ctrl,
                    preProcess = c("center", "scale"),
                    metric = "ROC")


#### ---- 6. Evaluación y Curvas ROC ---- ####

# Generación de probabilidades para el set de prueba
prob_svm_clean <- predict(svmClean, newdata = testData_clean, type = "prob")
prob_rf_clean  <- predict(rfClean,  newdata = testData_clean, type = "prob")
prob_knn_clean <- predict(knnClean, newdata = testData_clean, type = "prob")
prob_paper     <- predict(modelPaper, newdata = testData_paper, type = "prob")

# Definición de clase de interés: "M" (Maligno)
col_interes <- "M" 

# Cálculo de objetos ROC
roc_svm   <- roc(testData_clean$Diagnosis, prob_svm_clean[, col_interes], quiet = TRUE) 
roc_rf    <- roc(testData_clean$Diagnosis, prob_rf_clean[, col_interes], quiet = TRUE)
roc_knn   <- roc(testData_clean$Diagnosis, prob_knn_clean[, col_interes], quiet = TRUE)
roc_paper <- roc(testData_paper$Diagnosis, prob_paper[, col_interes], quiet = TRUE) 

# Extracción de valores AUC
auc_svm   <- auc(roc_svm)
auc_rf    <- auc(roc_rf)
auc_knn   <- auc(roc_knn)
auc_paper <- auc(roc_paper)

# Impresión de resultados comparativos
cat("\n--- Resultados de AUC (Área Bajo la Curva) ---\n")
cat("SVM Optimizado: ", round(auc_svm, 4), "\n")
cat("Random Forest:  ", round(auc_rf, 4), "\n")
cat("k-NN:           ", round(auc_knn, 4), "\n")
cat("Paper (3 vars): ", round(auc_paper, 4), "\n")

# Interpretación: SVM presenta el AUC más alto (0.998), seguido muy de cerca 
# por el modelo del Paper (0.997). Se seleccionan estos dos para la gráfica final.


#### ---- 7. Visualización Comparativa ---- ####

# Gráfico 1: Modelo SVM Optimizado
plot(roc_svm, 
     col = "pink", 
     main = "Comparativa: SVM Optimizado vs Paper Original",
     xlab = "Tasa de Falsos Positivos (1 - Especificidad)",
     ylab = "Tasa de Verdaderos Positivos (Sensibilidad)",
     lwd = 3, 
     legacy.axes = TRUE, 
     print.auc = FALSE)

# Gráfico 2: Modelo Paper (Superpuesto)
plot(roc_paper, 
     col = "blue", 
     add = TRUE, 
     lwd = 3, 
     lty = 2)

# Leyenda
legend("bottomright", 
       legend = c(paste0("SVM (21 vars) - AUC: ", round(auc_svm, 4)),
                  paste0("Paper (3 vars)    - AUC: ", round(auc_paper, 4))),
       col = c("pink", "blue"), 
       lwd = 3, 
       lty = c(1, 2), 
       bg = "white", 
       cex = 0.9)

grid()


#### ---- 8. Matrices de Confusión ---- ####

# Predicciones finales (Clases)
preds_svm_final <- predict(svmClean, newdata = testData_clean)
preds_paper_final <- predict(modelPaper, newdata = testData_paper)

cat("\n--- Matriz de Confusión: SVM Optimizado ---\n")
cm_svm <- confusionMatrix(preds_svm_final, testData_clean$Diagnosis, positive= "M")
print(cm_svm$table)

cat("\n--- Matriz de Confusión: Paper Original ---\n")
cm_paper <- confusionMatrix(preds_paper_final, testData_paper$Diagnosis, positive ="M")
print(cm_paper$table)

#### ---- 9. Conclusiones Finales ---- ####

# 1. Parsimonia: El modelo reducido (3 variables) logra una sensibilidad idéntica 
#    al modelo complejo, validando la literatura existente.
# 2. Especificidad: El modelo optimizado (21 variables) elimina los Falsos Positivos 
#    por completo en este conjunto de prueba.
# 3. Robustez: Ambos modelos superan el umbral de 0.99 AUC, confirmando la 
#    separabilidad lineal de los datos.
