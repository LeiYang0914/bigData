# ============================================
# PySpark Implementation - Heart Disease Analysis
# Multiple Models: Logistic Regression, Random Forest, GBT
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark import StorageLevel
import numpy as np
import random


# ============================================
# 1. Initialize Spark Session
# ============================================
spark = SparkSession.builder \
    .appName("HeartDiseaseAnalysis") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark version:", spark.version)

# ============================================
# 2. Load Dataset
# ============================================
df = spark.read.csv("gs://bd-bucket-01/heart_disease.csv",
                    header=True,
                    inferSchema=True)

# ============================================
# 3. Basic Dataset Information
# ============================================

# First rows
print("First 5 rows:")
df.show(5)

# Dimensions
print(f"Number of rows: {df.count()}")
print(f"Number of columns: {len(df.columns)}")

# Column names
print("Columns:", df.columns)

# Data types
print("Schema:")
df.printSchema()

# Summary statistics
print("Summary statistics:")
df.describe().show()

# ============================================
# 4. Missing Value Check
# ============================================

print("\nMissing values per column:")
for col in df.columns:
    null_count = df.filter(F.col(col).isNull()).count()
    print(f"{col}: {null_count}")

# ============================================
# 5. Duplicate Row Check
# ============================================

num_dups = df.count() - df.dropDuplicates().count()
print(f"\nNumber of duplicate rows: {num_dups}")

# ============================================
# 6. Distribution of Key Variables
# ============================================

print("\nHeartDiseaseorAttack counts:")
df.groupBy("HeartDiseaseorAttack").count().show()

print("\nBMI Summary:")
df.select("BMI").describe().show()

# ============================================
# 7. Data Preprocessing
# ============================================

df_clean = df

# 7.1 Drop missing values
rows_before = df_clean.count()
df_clean = df_clean.dropna()
rows_after = df_clean.count()
print(f"\nRows before dropping missing values: {rows_before}")
print(f"Rows after dropping missing values: {rows_after}")

# 7.2 Remove duplicates
df_clean = df_clean.dropDuplicates()
print(f"Rows after removing duplicates: {df_clean.count()}")

# 7.3 Handle BMI outliers (Winsorize 1-99 percentile)
quantiles = df_clean.approxQuantile("BMI", [0.01, 0.99], 0.01)
bmi_q_low, bmi_q_high = quantiles[0], quantiles[1]

df_clean = df_clean.withColumn(
    "BMI_capped",
    F.when(F.col("BMI") < bmi_q_low, bmi_q_low)
     .when(F.col("BMI") > bmi_q_high, bmi_q_high)
     .otherwise(F.col("BMI"))
)

print("\nBMI_capped summary:")
df_clean.select("BMI_capped").describe().show()

# 7.4 Recode GenHlth → String Factor
genhlth_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
genhlth_mapping = F.create_map([F.lit(x) for pair in genhlth_map.items() for x in pair])
df_clean = df_clean.withColumn("GenHlth_Factor", genhlth_mapping[F.col("GenHlth")])

# 7.5 Validate MentHlth & PhysHlth ranges (0-30)
invalid_ment = df_clean.filter(F.col("MentHlth") > 30).count()
invalid_phys = df_clean.filter(F.col("PhysHlth") > 30).count()

print(f"Rows with MentHlth > 30: {invalid_ment}")
print(f"Rows with PhysHlth > 30: {invalid_phys}")

df_clean = df_clean.filter((F.col("MentHlth") <= 30) & (F.col("PhysHlth") <= 30))

# 7.6 BMI Category
df_clean = df_clean.withColumn(
    "BMI_Category",
    F.when(F.col("BMI") < 18.5, "Underweight")
     .when(F.col("BMI") < 25, "Normal")
     .when(F.col("BMI") < 30, "Overweight")
     .otherwise("Obese")
)

# 7.7 AgeGroup mapping
age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44",
              "45-49", "50-54", "55-59", "60-64",
              "65-69", "70-74", "75-79", "80+"]

age_map = {i+1: label for i, label in enumerate(age_labels)}
age_mapping = F.create_map([F.lit(x) for pair in age_map.items() for x in pair])
df_clean = df_clean.withColumn("AgeGroup", age_mapping[F.col("Age").cast("int")])

# 7.8 AgeBand (broader groups)
df_clean = df_clean.withColumn(
    "AgeBand",
    F.when(F.col("Age") <= 3, "18-34")
     .when(F.col("Age") <= 6, "35-49")
     .when(F.col("Age") <= 9, "50-64")
     .when(F.col("Age") <= 11, "65-74")
     .otherwise("75+")
)

# 7.9 Lifestyle Risk Score
df_clean = df_clean.withColumn(
    "RiskScore",
    F.col("Smoker").cast("int") +
    F.col("HvyAlcoholConsump").cast("int") +
    (1 - F.col("PhysActivity").cast("int")) +
    (1 - F.col("Fruits").cast("int")) +
    (1 - F.col("Veggies").cast("int"))
)

# 7.10 Disease Burden Index
df_clean = df_clean.withColumn(
    "DiseaseCount",
    F.col("HighBP").cast("int") +
    F.col("HighChol").cast("int") +
    F.col("Diabetes").cast("int") +
    F.col("Stroke").cast("int")
)

# 7.11 Health Stress Index
df_clean = df_clean.withColumn(
    "HealthStressIndex",
    F.col("MentHlth") + F.col("PhysHlth")
)

# 7.12 Healthcare Access Score
df_clean = df_clean.withColumn(
    "HealthcareScore",
    F.col("AnyHealthcare").cast("int") +
    (1 - F.col("NoDocbcCost").cast("int"))
)

# 7.13 Obesity Indicator
df_clean = df_clean.withColumn(
    "ObeseFlag",
    F.when(F.col("BMI") >= 30, 1).otherwise(0)
)

# 7.14 Lifestyle Profile Category
df_clean = df_clean.withColumn(
    "LifestyleProfile",
    F.when(F.col("RiskScore") <= 1, "Healthy")
     .when(F.col("RiskScore") <= 3, "ModerateRisk")
     .otherwise("HighRisk")
)

print("\nPreprocessing complete!")
df_clean.select("Age", "AgeGroup", "AgeBand", "RiskScore", "LifestyleProfile").show(10)

# Save as single CSV file (for easy download)
output_path = "gs://bd-bucket-01/heart_disease_cleaned.csv"

print(f"\nSaving clean dataset to: {output_path}")

# Coalesce to 1 partition for single file output
df_clean.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)

print("Clean dataset saved successfully!")

# ============================================
# 8. Machine Learning Pipeline Setup
# ============================================

# Feature columns
numeric_features = ["Age", "PhysHlth", "MentHlth", "HealthStressIndex", 
                   "DiseaseCount", "ObeseFlag", "RiskScore", "BMI",
                   "Sex", "HighBP", "HighChol", "Diabetes", "Stroke",
                   "Smoker", "PhysActivity"]

categorical_features = ["AgeGroup", "AgeBand", "LifestyleProfile"]

# Target column
target_col = "HeartDiseaseorAttack"

# ============================================
# 8.1 Train/Val/Test Split (80/10/10)
# ============================================

# First split: 80% train, 20% temp
train_df, temp_df = df_clean.randomSplit([0.8, 0.2], seed=123)

# Second split: split temp into 50/50 for val and test
val_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=123)

print(f"\nTrain count: {train_df.count()}")
print(f"Validation count: {val_df.count()}")
print(f"Test count: {test_df.count()}")

# ============================================
# 8.2 Feature Engineering Pipelines (2 versions)
# ============================================

# ---------- Common: StringIndexer ----------
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
    for col in categorical_features
]

# ---------- Logistic Regression Pipeline (OHE + Scaling) ----------

# One-hot encode categorical columns
encoders = [
    OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec")
    for col in categorical_features
]
ohe_cat_cols = [col + "_vec" for col in categorical_features]

assembler_lr = VectorAssembler(
    inputCols=numeric_features + ohe_cat_cols,
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

lr_feature_stages = indexers + encoders + [assembler_lr, scaler]


# ---------- Tree Model Pipeline (Index Only, NO OHE, NO SCALING) ----------

assembler_tree = VectorAssembler(
    inputCols=numeric_features + [col + "_idx" for col in categorical_features],
    outputCol="features"
)

tree_feature_stages = indexers + [assembler_tree]

# ============================================
# 8.3 Evaluation Function
# ============================================

def evaluate_model(predictions, dataset_name="Test"):
    """Evaluate model predictions and print metrics"""
    
    # Binary Classification Evaluator (AUC)
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=target_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Multiclass Evaluators
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="accuracy"
    )
    
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="weightedPrecision"
    )
    
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="weightedRecall"
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="f1"
    )
    
    # Calculate metrics
    auc = evaluator_auc.evaluate(predictions)
    acc = evaluator_acc.evaluate(predictions)
    prec = evaluator_prec.evaluate(predictions)
    rec = evaluator_rec.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    print(f"\n================ {dataset_name} RESULTS =================")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    
    # Confusion matrix
    print(f"\n{dataset_name} Confusion Matrix:")
    predictions.groupBy(target_col, "prediction").count().orderBy(target_col, "prediction").show()
    
    return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ============================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================

print("\n" + "="*70)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*70)

# Create Logistic Regression model
lr = LogisticRegression(
    labelCol=target_col,
    featuresCol="features",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5,
    family="binomial"
)

# Build pipeline
lr_pipeline = Pipeline(stages=lr_feature_stages + [lr])

# Train model
print("\nTraining Logistic Regression model...")
lr_model = lr_pipeline.fit(train_df)

# Make predictions
lr_train_pred = lr_model.transform(train_df)
lr_val_pred = lr_model.transform(val_df)
lr_test_pred = lr_model.transform(test_df)

# Evaluate
lr_train_metrics = evaluate_model(lr_train_pred, "Logistic Regression - Train")
lr_val_metrics = evaluate_model(lr_val_pred, "Logistic Regression - Validation")
lr_test_metrics = evaluate_model(lr_test_pred, "Logistic Regression - Test")


# ============================================
# MODEL 2: RANDOM FOREST
# ============================================

print("\n" + "="*70)
print("MODEL 2: RANDOM FOREST")
print("="*70)

# Create Random Forest model
rf = RandomForestClassifier(
    labelCol=target_col,
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=5,
    seed=123
)

# Build pipeline
rf_pipeline = Pipeline(stages=tree_feature_stages + [rf])

# Train model
print("\nTraining Random Forest model...")
rf_model = rf_pipeline.fit(train_df)

# Make predictions
rf_train_pred = rf_model.transform(train_df)
rf_val_pred = rf_model.transform(val_df)
rf_test_pred = rf_model.transform(test_df)

# Evaluate
rf_train_metrics = evaluate_model(rf_train_pred, "Random Forest - Train")
rf_val_metrics = evaluate_model(rf_val_pred, "Random Forest - Validation")
rf_test_metrics = evaluate_model(rf_test_pred, "Random Forest - Test")

# ============================================
# MODEL 3: GRADIENT BOOSTED TREES (GBT)
# ============================================

print("\n" + "="*70)
print("MODEL 3: GRADIENT BOOSTED TREES (GBT)")
print("="*70)

# Create GBT model
gbt = GBTClassifier(
    labelCol=target_col,
    featuresCol="features",
    maxDepth=6,
    maxIter=100,
    stepSize=0.1,
    seed=123
)

# Build pipeline
gbt_pipeline = Pipeline(stages=tree_feature_stages + [gbt])

# Train model
print("\nTraining GBT model...")
gbt_model = gbt_pipeline.fit(train_df)

# Make predictions
gbt_train_pred = gbt_model.transform(train_df)
gbt_val_pred = gbt_model.transform(val_df)
gbt_test_pred = gbt_model.transform(test_df)

# Evaluate
gbt_train_metrics = evaluate_model(gbt_train_pred, "GBT - Train")
gbt_val_metrics = evaluate_model(gbt_val_pred, "GBT - Validation")
gbt_test_metrics = evaluate_model(gbt_test_pred, "GBT - Test")

# ============================================
# MODEL COMPARISON SUMMARY
# ============================================

print("\n" + "="*70)
print("MODEL COMPARISON - TEST SET PERFORMANCE")
print("="*70)

comparison_data = [
    ("Logistic Regression", lr_test_metrics),
    ("Random Forest", rf_test_metrics),
    ("Gradient Boosted Trees", gbt_test_metrics)
]

print(f"\n{'Model':<25} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 75)
for model_name, metrics in comparison_data:
    print(f"{model_name:<25} {metrics['auc']:<10.4f} {metrics['accuracy']:<10.4f} "
          f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

# Find best model by AUC
best_model = max(comparison_data, key=lambda x: x[1]['auc'])
print(f"\nBest Model by AUC: {best_model[0]} (AUC = {best_model[1]['auc']:.4f})")
# ============================================
# SAVE BEST MODEL
# ============================================

print("\n" + "="*70)
print("SAVING THE BEST MODEL")
print("="*70)

# Map model names to model objects
model_objects = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "Gradient Boosted Trees": gbt_model
}

# Select the actual model object
best_model_name = best_model[0]
best_model_obj = model_objects[best_model_name]

print(f"Best model selected: {best_model_name}")

# Choose save path
model_path = f"gs://bd-bucket-01/models/best_model_{best_model_name.replace(' ', '_').lower()}"

try:
    best_model_obj.write().overwrite().save(model_path)
    print(f"Model saved successfully at: {model_path}")
except Exception as e:
    print(f"Could not save model. Error: {str(e)}")
    print("Check GCS bucket permissions for write access.")


# Stop Spark session
spark.stop()