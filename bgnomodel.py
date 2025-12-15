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


# Save splits
print("\nSaving train/val/test splits...")

train_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("gs://bd-bucket-01/splits/train.csv")

val_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("gs://bd-bucket-01/splits/val.csv")

test_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("gs://bd-bucket-01/splits/test.csv")

print("Train/val/test splits saved!")
print("\nDownload splits with:")
print("  gsutil cp gs://bd-bucket-01/splits/train.csv/*.csv ./train.csv")
print("  gsutil cp gs://bd-bucket-01/splits/val.csv/*.csv ./val.csv")
print("  gsutil cp gs://bd-bucket-01/splits/test.csv/*.csv ./test.csv")

# Stop Spark session
spark.stop()