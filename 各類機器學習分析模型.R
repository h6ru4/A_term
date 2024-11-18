#==============================
# A. XGBoost 分析
#==============================

#==============================
# 1. 安裝並載入必要的套件
#==============================
packageName <- c("xgboost", "Matrix", "caret", "data.table", "dplyr","ICEbox")
for (i in 1:length(packageName)) {
  if (!(packageName[i] %in% rownames(installed.packages()))) {
    install.packages(packageName[i])
  }
}
library(xgboost)
library(Matrix)
library(caret)
library(data.table)
library(dplyr)
library(ICEbox)


#==============================
# 2. 讀取並處理原始數據集
#==============================
# 讀取原始數據集
setwd("D:/慈濟/Week4")  # 設定工作目錄
data <- read.csv("StudentPerformanceFactors.csv", header = TRUE)

# 檢查數據結構
str(data)

# 將所有 character 型和 factor 型變數轉換為因子
data <- data %>%
  mutate(across(where(is.character), as.factor))  # 將所有文字型資料轉為因子型

#==============================
# 3. 使用 dummyVars 進行 One-Hot 編碼
#==============================
# 使用 caret 的 dummyVars 進行 One-Hot 編碼，設置 fullRank = FALSE 來保留所有類別
dummy_model <- dummyVars(~ ., data = data, fullRank = FALSE)
data_one_hot <- predict(dummy_model, newdata = data)

# 檢查 One-Hot 編碼後的結果
str(data_one_hot)  # 應該包含所有類別的編碼

#==============================
# 4. 提取應變數與特徵變數
#==============================
# 假設最後一列為應變數，其他為特徵變數
target_index <- ncol(data_one_hot)
target <- data_one_hot[, target_index]
features <- data_one_hot[, -target_index]

# 檢查應變數與特徵變數
str(features)
str(target)

#==============================
# 5. 訓練集和測試集分割
#==============================
set.seed(123)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)

# 分割訓練集和測試集
trainData <- features[trainIndex, ]
testData <- features[-trainIndex, ]
print(colnames(trainData))
# 提取訓練集和測試集的標籤
trainLabel <- target[trainIndex]
testLabel <- target[-trainIndex]

write.csv(trainData,file ="訓練集.csv")
write.csv(testData,file ="測試集.csv")


#==============================
# 6. 交叉驗證設定
#==============================
# 使用 caret 進行交叉驗證來選擇最佳參數
# 5 折交叉驗證 在交叉驗證中（如 5 折交叉驗證），
# 每次劃分訓練集和驗證集，計算每次的 RMSE，然後求取平均值。

train_control <- trainControl(method = "cv", number = 5)  


# 定義調參範圍
tune_grid <- expand.grid(
  nrounds = c(50, 100, 150),   # 樹的輪數 : 建議範圍：100 到 1000
  max_depth = c(3, 6, 9),      # 樹的深度: 建議範圍：3 到 10。
  eta = c(0.01, 0.1, 0.3),     # 學習率
  gamma = 0,                   # 分裂節點的最小損失函數減少: c(0, 1, 5)
  colsample_bytree = 1,        # 每棵樹隨機選取的特徵比例: c(0.5, 0.8, 1)
  min_child_weight = 1,        # 最小葉子節點權重和: c(1, 5, 10)
  subsample = 1                # 每棵樹的樣本比例: c(0.5, 0.8, 1) 。較大的值（如 1）代表使用所有樣本
)

# 使用 caret 的 train 進行交叉驗證
set.seed(123)
xgb_cv <- train(
  x = trainData,
  y = trainLabel,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE"  # 使用 RMSE 作為評估標準
)

# 顯示最佳參數和平均驗證集 RMSE
# print(xgb_cv$results)
# print(paste("Average Valid RMSE:", mean(xgb_cv$results$RMSE)))


#==============================
# 7. 使用最佳參數訓練模型
#==============================
# 使用最佳參數設置進行模型訓練
best_params <- list(
  objective = "reg:squarederror",  # 迴歸目標
  eval_metric = "rmse",            # 評估指標
  eta = xgb_cv$bestTune$eta,       # 最佳學習率
  max_depth = xgb_cv$bestTune$max_depth,  # 最佳深度
  subsample = xgb_cv$bestTune$subsample,  # 最佳子樣本比例
  colsample_bytree = xgb_cv$bestTune$colsample_bytree  # 最佳特徵比例
)

# 訓練最終模型
xgbModel <- xgb.train(
  params = best_params,
  data = xgb.DMatrix(data = trainData, label = trainLabel),
  nrounds = xgb_cv$bestTune$nrounds
)



#==============================
# 8. 使用測試集進行預測與評估
#==============================
# 使用測試集進行預測
pred <- predict(xgbModel, xgb.DMatrix(data = testData))

result <- cbind(testData,testLabel, pred)
write.csv(result,file ="顯示結果_測試集vs預測.csv")

# 計算預測 RMSE
rmse <- sqrt(mean((pred - testLabel) ^ 2))
cat("測試集 RMSE:", rmse, "\n")

# 手動計算 R²
r2_manual <- 1 - sum((testLabel - pred)^2) / sum((testLabel - mean(testLabel))^2)
cat("Manual R-Squared (R²):", r2_manual, "\n")

#==============================
# 9. 儲存模型及變數名稱
#==============================
# 儲存模型
xgb.save(xgbModel, "xgb_student_performance_regression_model.model")

# 儲存變數名稱以便與新測試集對應
feature_info <- data.frame(feature_name = colnames(features))
write.csv(feature_info, "training_feature_info.csv", row.names = FALSE)

#==============================
# 10. 特徵重要性分析
#==============================
# 獲取特徵重要性
importance_matrix <- xgb.importance(feature_names = colnames(trainData), model = xgbModel)

# 顯示特徵重要性
print(importance_matrix)

# 繪製特徵重要性圖
xgb.plot.importance(importance_matrix)

# 保存特徵重要性到 CSV 檔案
write.csv(importance_matrix, file = "xgb_feature_importance_student_performance.csv", row.names = FALSE)

##############################################
# 使用訓練好的模型對新測試集進行預處理和預測 #
##############################################

#==============================
# 1. 讀取訓練好的模型和變數信息
#==============================
# 載入訓練好的模型
model_path <- "xgb_student_performance_regression_model.model"
xgbModel <- xgb.load(model_path)

# 讀取變數名稱信息
feature_info <- read.csv("training_feature_info.csv")
feature_names <- feature_info$feature_name

#==============================
# 2. 讀取新測試集並預處理
#==============================
# 讀取新測試集
new_test_data <- read.csv("新測試集.csv", header = TRUE)

# 檢查變數的層次數量，確保每個變數至少有兩個層次
for (col in colnames(new_test_data)) {
  if (is.factor(new_test_data[[col]]) | is.character(new_test_data[[col]])) {
    # 檢查層次數量
    levels_count <- length(unique(new_test_data[[col]]))
    if (levels_count < 2) {
      # 為只有一個層次的變數添加另一個層次
      new_test_data[[col]] <- factor(new_test_data[[col]], levels = c(unique(new_test_data[[col]]), "Other"))
    }
  }
}

# 使用 dummyVars 進行 One-Hot 編碼
dummy_model_new <- dummyVars(~ ., data = new_test_data, fullRank = FALSE)
new_test_data_encoded <- predict(dummy_model_new, newdata = new_test_data)

# 檢查 new_test_data_encoded 是否為數據框格式
if (!is.data.frame(new_test_data_encoded)) {
  new_test_data_encoded <- as.data.frame(new_test_data_encoded)
}

# 檢查 new_test_data_encoded 的結構和列數
cat("New Test Data Encoded Structure:\n")
#str(new_test_data_encoded)
cat("Column Count of Encoded Data:", ncol(new_test_data_encoded), "\n")

# 確保新測試集的列順序與訓練集一致
missing_vars <- setdiff(feature_names, colnames(new_test_data_encoded))
if (length(missing_vars) > 0) {
  for (var in missing_vars) {
    new_test_data_encoded[[var]] <- 0  # 添加缺失的變數列並賦值為 0
  }
}

# 確保新測試集包含所有訓練集變數
extra_vars <- setdiff(colnames(new_test_data_encoded), feature_names)
if (length(extra_vars) > 0) {
  new_test_data_encoded <- new_test_data_encoded[, feature_names]
}

# 檢查處理後的 new_test_data_encoded 結構
cat("Processed Test Data Encoded Structure:\n")
#str(new_test_data_encoded)

#==============================
# 3. 使用訓練好的模型進行預測
#==============================
# 使用模型對新測試集進行預測
dtest_new <- xgb.DMatrix(data = as.matrix(new_test_data_encoded))
new_predictions <- predict(xgbModel, dtest_new)

# 將預測結果加入到新測試集中
new_test_data$Predicted_Result <- new_predictions

#==============================
# 4. 保存預測結果
#==============================
# 保存預測結果至 CSV 文件
write.csv(new_test_data, "新測試集_預測結果.csv", row.names = FALSE)


##################################
#  顯示最後模型分析的效果        #
##################################

###  計算 R² （判定係數）: 衡量模型對於總變異的解釋能力，越接近 1 表示模型解釋能力越強。
r2_manual <- 1 - sum((testLabel - pred)^2) / sum((testLabel - mean(testLabel))^2)
cat("Manual R-Squared (R²):", r2_manual, "\n")
r2_rounded <- round(r2_manual, 4)
# 實際值與預測值的關係圖
plot(testLabel, pred, main =paste( "分析測試集：實際分數 vs. 預測分數",", R²=",r2_rounded), xlab = "實際分數", ylab = "預測分數")
abline(0, 1, col = "red",lwd=2)



### ICE（Individual Conditional Expectation）個別條件期望圖用來展示某一特徵在不同取值時，對個體預測結果的影響。

# 構建 DMatrix 格式數據
trainMatrix <- as.matrix(trainData) # 去掉目標變數列
dtrain <- xgb.DMatrix(data = trainMatrix, label = trainLabel)


# 提取特徵重要性
importance_matrix <- xgb.importance(feature_names = colnames(trainMatrix), model = xgbModel)

# 查看特徵重要性排名前 5 的特徵
top_features <- head(importance_matrix$Feature, 5)
print(top_features)

# 設置保存 ICE 圖的目錄
save_dir <- "ICE_plots"
if (!dir.exists(save_dir)) dir.create(save_dir)

# 循環繪製並保存每個重要特徵的 ICE 圖
for (feature in top_features) {
  # 構建 ICE 物件
  ice_obj <- ice(object = xgbModel, X = trainMatrix, y = trainLabel, predictor = feature)
  
  # 設置保存路徑和文件名
  file_name <- file.path(save_dir, paste0("ICE_Plot_", feature, ".png"))
  
  # 打開 PNG 圖形設備
  png(filename = file_name, width = 800, height = 600)
  
  # 繪製 ICE 圖
  plot(ice_obj, main = paste("ICE Plot for", feature))
  
  # 關閉圖形設備，保存文件
  dev.off()
  
  cat("Saved ICE plot for", feature, "to", file_name, "\n")
}








#==============================
#  B. 線性回歸模型
#==============================
packageName <- c("rpart", "rpart.plot", "randomForest", "FactoMineR","glmnet")
for (i in 1:length(packageName)) {
  if (!(packageName[i] %in% rownames(installed.packages()))) {
    install.packages(packageName[i])
  }
}


# 加載所需的庫
library(rpart)
library(rpart.plot)
library(randomForest)
library(FactoMineR)


# 將分類變數轉換為因子型變數
# 說明：將分類變數（如'Extracurricular_Activities'）轉換為因子，以便後續建模
data$Extracurricular_Activities <- as.factor(data$Extracurricular_Activities)
data$Internet_Access <- as.factor(data$Internet_Access)
data$School_Type <- as.factor(data$School_Type)
data$Learning_Disabilities <- as.factor(data$Learning_Disabilities)
data$Gender <- as.factor(data$Gender)

# 線性回歸模型，包含所有自變數
lm_model <- lm(Exam_Score ~ ., data = data)

# 檢查線性回歸模型摘要
summary(lm_model)  

# 檢查線性回歸模型摘要
model_summary <- summary(lm_model)

# 提取係數和p值
coefficients <- model_summary$coefficients

# 將係數和p值轉換為數據框
coeff_df <- as.data.frame(coefficients)

# 添加 Signif. codes 註記
coeff_df$`Signif. codes` <- cut(coeff_df[, 4],
                                breaks = c(-Inf, 0.001, 0.01, 0.05, 0.1, Inf),
                                labels = c("***", "**", "*", ".", " "),
                                 right = FALSE)
 
# ***：p值 < 0.001，非常顯著。
# **：p值 < 0.01，顯著。
# *：p值 < 0.05，弱顯著。
# .：p值 < 0.1，邊緣顯著。
# ：p值 >= 0.1，無顯著性。


# 按p值排序
coeff_df_sorted <- coeff_df[order(coeff_df[, 4]), ]

# 顯示按p值排序並附加Signif. codes的結果
print(coeff_df_sorted)

# 將排序後的結果寫入csv檔案
write.csv(coeff_df_sorted, file = "lm_model_sorted_significance.csv", row.names = TRUE)

# 印提示信息
print("結果已保存為 'lm_model_sorted_significance.csv'")








#==============================
#  C. 邏輯回歸模型
#==============================
# 說明：將Exam_Score二元化（如將70分設為界限），然後使用邏輯回歸檢查分類變數對結果的影響
data$Exam_Result <- ifelse(data$Exam_Score > 70, 1, 0)


# 選擇一些較重要的變數進行邏輯迴歸
logit_model <- glm(Exam_Result ~ 
                  + Hours_Studied
                  + Attendance
                  + Access_to_Resources
                  + Previous_Scores
                  + Parental_Involvement
                  + Tutoring_Sessions
                  + Parental_Involvement
                  + Access_to_Resources
                  + Family_Income
                  + Peer_Influence
                  + Motivation_Level
                  + Extracurricular_Activities
                  + Learning_Disabilities
                  + Internet_Access
                  + Family_Income
                  + Motivation_Level
                  + Physical_Activity
                  + Peer_Influence
                  + Parental_Education_Level
                  ,family = binomial, data = data)
summary(logit_model)

# 檢查邏輯迴歸模型摘要
logit_summary <- summary(logit_model)

# 提取係數和p值
logit_coefficients <- logit_summary$coefficients

# 將係數和p值轉換為數據框
logit_coeff_df <- as.data.frame(logit_coefficients)

# 添加 Signif. codes 註記
logit_coeff_df$`Signif. codes` <- cut(logit_coeff_df[, 4],
                                      breaks = c(-Inf, 0.001, 0.01, 0.05, 0.1, Inf),
                                      labels = c("***", "**", "*", ".", " "),
                                      right = FALSE)

# 按p值排序
logit_coeff_df_sorted <- logit_coeff_df[order(logit_coeff_df[, 4]), ]

# 將排序後的結果寫入csv檔案
write.csv(logit_coeff_df_sorted, file = "logit_model_sorted_significance.csv", row.names = TRUE)

# 打印提示信息
print("結果已保存為 'logit_model_sorted_significance.csv'")



#==============================
#  D. 相關性分析
#==============================
# 說明：計算連續變數之間的相關性（例如`Hours_Studied`與`Exam_Score`的相關性）
# 提取所有連續變數
if(!require(Hmisc)) install.packages("Hmisc", dependencies = TRUE)
library(Hmisc)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 提取所有連續變數
continuous_vars <- data[, sapply(data, is.numeric)]

# 計算相關性矩陣和p值矩陣
cor_results <- rcorr(as.matrix(continuous_vars))

# 提取相關係數和p值
cor_matrix <- cor_results$r
p_matrix <- cor_results$P

# 創建帶有顯著性標註的矩陣
signif_codes <- ifelse(p_matrix < 0.001, "***", 
                       ifelse(p_matrix < 0.01, "**", 
                              ifelse(p_matrix < 0.05, "*", 
                                     ifelse(p_matrix < 0.1, ".", " "))))

# 將相關性矩陣轉換為數據框
cor_df <- as.data.frame(cor_matrix)

# 添加顯著性符號
for (i in 1:nrow(cor_matrix)) {
  for (j in 1:ncol(cor_matrix)) {
    cor_df[i, j] <- paste0(round(cor_matrix[i, j], 3), signif_codes[i, j])
  }
}

# 打印相關性矩陣及顯著性符號
print(cor_df)

# 將帶有顯著性標註的相關性矩陣寫入CSV文件
write.csv(cor_df, file = "continuous_vars_correlation_with_significance.csv", row.names = TRUE)

# 顯示提示信息
print("相關性矩陣及顯著性標註已保存為 'continuous_vars_correlation_with_significance.csv'")



#==============================
#  E. 決策樹分析
#==============================
# 說明：使用決策樹模型來直觀地展示不同變數對`Exam_Score`的影響，並進行視覺化
# 加載必要的包
if (!require(rpart)) install.packages("rpart")
if (!require(rpart.plot)) install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 構建決策樹模型
tree_model <- rpart(Exam_Score ~ ., data = data, method = "anova")

# 保存決策樹圖為PNG文件
png("decision_tree_plot.png", width = 800, height = 600)
rpart.plot(tree_model, type = 3, digits = 3)
dev.off()

# 顯示提示信息
print("決策樹圖已保存為 'decision_tree_plot.png'")



#==============================
#  F. 隨機森林模型
#==============================
# 說明：使用隨機森林模型來分析數據，並顯示變數的重要性
# 加載必要的包
if (!require(randomForest)) install.packages("randomForest")

library(randomForest)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 檢查並處理分類變數 (將它們轉換為因子)
data$Extracurricular_Activities <- as.factor(data$Extracurricular_Activities)
data$Internet_Access <- as.factor(data$Internet_Access)
data$School_Type <- as.factor(data$School_Type)
data$Learning_Disabilities <- as.factor(data$Learning_Disabilities)
data$Gender <- as.factor(data$Gender)

# 使用隨機森林模型來預測Exam_Score
set.seed(123)  # 設定隨機種子以確保結果可重複
rf_model <- randomForest(Exam_Score ~ ., data = data, ntree = 500, importance = TRUE)

# 顯示變數重要性並保存為PNG文件
png("variable_importance_rf.png", width = 800, height = 600)
varImpPlot(rf_model)
dev.off()

# 顯示提示信息
print("變數重要性圖已保存為 'variable_importance_rf.png'")



#==============================
#  G. 主成分分析 (PCA)
#==============================
# 說明：主成分分析（PCA）有助於減少維度，找出對`Exam_Score`影響最大的主要成分
# 加載必要的包
if (!require(FactoMineR)) install.packages("FactoMineR")

library(FactoMineR)
library(factoextra)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 提取所有連續變數進行主成分分析
continuous_vars <- data[, sapply(data, is.numeric)]

# 執行PCA分析
pca_result <- PCA(continuous_vars, graph = FALSE)

# 保存主成分分析變數圖為PNG文件
png("pca_variables_plot.png", width = 800, height = 600)
fviz_pca_var(pca_result, repel = TRUE)
dev.off()

# 顯示提示信息
print("PCA變數圖已保存為 'pca_variables_plot.png'")


#==============================
#  H. 交互效應分析
#==============================
# 說明：檢查`Parental_Involvement`與`Parental_Education_Level`之間的交互效應，了解它們共同影響`Exam_Score`的情況
interaction_model <- lm(Exam_Score ~ Parental_Involvement * Parental_Education_Level, data = data)
summary(interaction_model)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 將分類變數轉換為因子
data$Parental_Involvement <- as.factor(data$Parental_Involvement)
data$Parental_Education_Level <- as.factor(data$Parental_Education_Level)

# 構建交互效應模型，檢查`Parental_Involvement`與`Parental_Education_Level`之間的交互效應
interaction_model <- lm(Exam_Score ~ Parental_Involvement * Parental_Education_Level, data = data)

# 獲取模型摘要
model_summary <- summary(interaction_model)

# 打印模型摘要
print(model_summary)

# 保存模型摘要為文本文件
capture.output(model_summary, file = "interaction_effect_summary.txt")

# 顯示提示信息
print("交互效應模型的結果已保存為 'interaction_effect_summary.txt'")




#==============================
#  I. 支持向量機（SVM）分析
#==============================
# 支持向量機（SVM） 這裡假設將Exam_Score轉換為二元分類問題，將成績分為及格（>=60）和不及格（<60）
# 安裝並加載必要的包
if (!require(e1071)) install.packages("e1071")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(cowplot)) install.packages("cowplot")

library(e1071)
library(ggplot2)
library(cowplot)

# 讀取數據集
data <- read.csv("StudentPerformanceFactors.csv")

# 檢查數據中的缺失值
print("缺失值檢查：")
print(colSums(is.na(data)))

# 檢查 Hours_Studied 和 Attendance 是否存在數值數據
print("Hours_Studied 檢查：")
print(table(data$Hours_Studied))

print("Attendance 檢查：")
print(table(data$Attendance))

# 處理數據中的缺失值（例如移除缺失值行）
data <- na.omit(data)

# 將Exam_Score分成二元類別，例如及格(>=60)或不及格(<60)
data$Exam_Result <- ifelse(data$Exam_Score >= 60, 1, 0)
data$Exam_Result <- as.factor(data$Exam_Result)

# 簡化為兩個連續變數進行分類
# 使用學習時間和出勤率進行可視化
svm_data <- data[, c("Hours_Studied", "Attendance", "Exam_Result")]

# 檢查變數是否仍有有效數據
if (nrow(svm_data) == 0) {
  stop("數據中沒有可用的 Hours_Studied 和 Attendance 變數進行SVM分析。請檢查數據。")
}

# 確保數據範圍有效
x_min <- min(svm_data$Hours_Studied, na.rm = TRUE) - 1
x_max <- max(svm_data$Hours_Studied, na.rm = TRUE) + 1
y_min <- min(svm_data$Attendance, na.rm = TRUE) - 1
y_max <- max(svm_data$Attendance, na.rm = TRUE) + 1

# 構建SVM模型
set.seed(123)  # 設置隨機種子以便結果可重現
svm_model <- svm(Exam_Result ~ Hours_Studied + Attendance, data = svm_data, kernel = "linear", scale = TRUE)

# 創建網格以便繪製分類邊界
grid <- expand.grid(Hours_Studied = seq(x_min, x_max, length.out = 100),
                    Attendance = seq(y_min, y_max, length.out = 100))

# 預測網格點的分類結果
grid$Exam_Result <- predict(svm_model, grid)

# 繪製SVM分類邊界
p <- ggplot(svm_data, aes(x = Hours_Studied, y = Attendance, color = Exam_Result)) +
  geom_point(size = 2) + 
  geom_point(data = grid, aes(x = Hours_Studied, y = Attendance, color = Exam_Result), alpha = 0.3, size = 0.5) +
  stat_contour(data = grid, aes(x = Hours_Studied, y = Attendance, z = as.numeric(Exam_Result)), 
               breaks = 0.5, color = "black") + 
  ggtitle("SVM 分類結果與邊界") +
  theme_minimal()

# 顯示SVM分類邊界圖
print(p)

# 保存圖形為PNG文件
ggsave("svm_classification_plot.png", plot = p, width = 8, height = 6)



# 顯示提示信息
print("SVM模型的結果已保存為 'svm_classification_plot.png'")

