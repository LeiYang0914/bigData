# ============================================
# PART 2: RUN THIS ON LOCAL CLOUDERA
# Hyperparameter Tuning with Pre-split Train/Val/Test CSVs
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# ============================================
# 1. Initialize Spark (Local Mode)
# ============================================
spark = SparkSession.builder \
    .appName("HeartDisease_LocalTuning") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

print("Spark version:", spark.version)

# ============================================
# 2. Load Pre-split Train/Val/Test CSVs
# ============================================

print("\nLoading datasets...")

# IMPORTANT: Put your CSV files in the same directory as this script
# Or provide full paths to the CSV files

train_df = spark.read.csv("train.csv", header=True, inferSchema=True)
val_df = spark.read.csv("val.csv", header=True, inferSchema=True)
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)

print(f"Train dataset: {train_df.count()} rows, {len(train_df.columns)} columns")
print(f"Validation dataset: {val_df.count()} rows, {len(val_df.columns)} columns")
print(f"Test dataset: {test_df.count()} rows, {len(test_df.columns)} columns")

# ============================================
# 3. OPTIONAL: Sample Data for Memory Management
# ============================================

# If your local machine has limited memory, you can sample the training data
USE_SAMPLING = False  # Set to True if you need to reduce memory usage
SAMPLE_FRACTION = 0.5  # Use 50% of training data

if USE_SAMPLING:
    print(f"\nSampling {SAMPLE_FRACTION*100}% of training data to reduce memory usage...")
    train_df = train_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=123)
    print(f"Sampled train dataset: {train_df.count()} rows")

# Cache datasets for faster access during Cross-Validation
print("\nCaching datasets for faster access...")
train_df.cache()
val_df.cache()
test_df.cache()

# Force cache by counting
train_df.count()
val_df.count()
test_df.count()

print("Datasets cached successfully!")

# ============================================
# 4. Feature Configuration
# ============================================

numeric_features = ["Age", "PhysHlth", "MentHlth", "HealthStressIndex", 
                   "DiseaseCount", "ObeseFlag", "RiskScore", "BMI",
                   "Sex", "HighBP", "HighChol", "Diabetes", "Stroke",
                   "Smoker", "PhysActivity"]

categorical_features = ["AgeGroup", "AgeBand", "LifestyleProfile"]
target_col = "HeartDiseaseorAttack"

# String Indexers
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
    for col in categorical_features
]

# Evaluator
evaluator = BinaryClassificationEvaluator(
    labelCol=target_col, 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderROC"
)

# ============================================
# 5. MODEL 1: LOGISTIC REGRESSION TUNING
# ============================================

print("\n" + "="*70)
print("TUNING MODEL 1: LOGISTIC REGRESSION")
print("="*70)

# Feature pipeline (with OHE and Scaling)
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

lr = LogisticRegression(
    labelCol=target_col,
    featuresCol="features",
    maxIter=100,
    family="binomial"
)

lr_pipeline = Pipeline(stages=indexers + encoders + [assembler_lr, scaler, lr])

# Parameter Grid
lr_param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

print(f"Total LR combinations to try: {len(lr_param_grid)}")

# CrossValidator
lr_cv = CrossValidator(
    estimator=lr_pipeline,
    estimatorParamMaps=lr_param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,
    seed=123
)

print("Training Logistic Regression with CrossValidation...")
print("(This may take a few minutes...)")
lr_cv_model = lr_cv.fit(train_df)

# Best model and parameters
best_lr_model = lr_cv_model.bestModel
lr_stage = best_lr_model.stages[-1]

print("\n--- Best Logistic Regression Parameters ---")
print(f"regParam: {lr_stage.getRegParam()}")
print(f"elasticNetParam: {lr_stage.getElasticNetParam()}")

# Validation evaluation
lr_val_pred = lr_cv_model.transform(val_df)
lr_val_auc = evaluator.evaluate(lr_val_pred)

# Additional metrics
acc_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
prec_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="weightedPrecision")
rec_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="weightedRecall")
f1_eval = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="f1")

lr_val_acc = acc_eval.evaluate(lr_val_pred)
lr_val_prec = prec_eval.evaluate(lr_val_pred)
lr_val_rec = rec_eval.evaluate(lr_val_pred)
lr_val_f1 = f1_eval.evaluate(lr_val_pred)

print(f"\n--- Validation Metrics ---")
print(f"AUC      : {lr_val_auc:.4f}")
print(f"Accuracy : {lr_val_acc:.4f}")
print(f"Precision: {lr_val_prec:.4f}")
print(f"Recall   : {lr_val_rec:.4f}")
print(f"F1 Score : {lr_val_f1:.4f}")

# ============================================
# 6. MODEL 2: RANDOM FOREST TUNING
# ============================================

print("\n" + "="*70)
print("TUNING MODEL 2: RANDOM FOREST")
print("="*70)

# Feature pipeline (Index only, no OHE/Scaling)
assembler_tree = VectorAssembler(
    inputCols=numeric_features + [col + "_idx" for col in categorical_features],
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol=target_col,
    featuresCol="features",
    seed=123
)

rf_pipeline = Pipeline(stages=indexers + [assembler_tree, rf])

# Parameter Grid
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

print(f"Total RF combinations to try: {len(rf_param_grid)}")

rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,
    seed=123
)

print("Training Random Forest with CrossValidation...")
print("(This may take several minutes...)")
rf_cv_model = rf_cv.fit(train_df)

# Best model and parameters
best_rf_model = rf_cv_model.bestModel
rf_stage = best_rf_model.stages[-1]

print("\n--- Best Random Forest Parameters ---")
print(f"numTrees: {rf_stage.getNumTrees}")
print(f"maxDepth: {rf_stage.getMaxDepth()}")
print(f"minInstancesPerNode: {rf_stage.getMinInstancesPerNode()}")

# Validation evaluation
rf_val_pred = rf_cv_model.transform(val_df)
rf_val_auc = evaluator.evaluate(rf_val_pred)
rf_val_acc = acc_eval.evaluate(rf_val_pred)
rf_val_prec = prec_eval.evaluate(rf_val_pred)
rf_val_rec = rec_eval.evaluate(rf_val_pred)
rf_val_f1 = f1_eval.evaluate(rf_val_pred)

print(f"\n--- Validation Metrics ---")
print(f"AUC      : {rf_val_auc:.4f}")
print(f"Accuracy : {rf_val_acc:.4f}")
print(f"Precision: {rf_val_prec:.4f}")
print(f"Recall   : {rf_val_rec:.4f}")
print(f"F1 Score : {rf_val_f1:.4f}")

# ============================================
# 7. MODEL 3: GRADIENT BOOSTED TREES TUNING
# ============================================

print("\n" + "="*70)
print("TUNING MODEL 3: GRADIENT BOOSTED TREES")
print("="*70)

gbt = GBTClassifier(
    labelCol=target_col,
    featuresCol="features",
    seed=123
)

gbt_pipeline = Pipeline(stages=indexers + [assembler_tree, gbt])

# Parameter Grid
gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [4, 6, 8]) \
    .addGrid(gbt.maxIter, [50, 100, 150]) \
    .addGrid(gbt.stepSize, [0.05, 0.1, 0.2]) \
    .build()

print(f"Total GBT combinations to try: {len(gbt_param_grid)}")

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,
    seed=123
)

print("Training GBT with CrossValidation...")
print("(This may take several minutes...)")
gbt_cv_model = gbt_cv.fit(train_df)

# Best model and parameters
best_gbt_model = gbt_cv_model.bestModel
gbt_stage = best_gbt_model.stages[-1]

print("\n--- Best GBT Parameters ---")
print(f"maxDepth: {gbt_stage.getMaxDepth()}")
print(f"maxIter: {gbt_stage.getMaxIter()}")
print(f"stepSize: {gbt_stage.getStepSize()}")

# Validation evaluation
gbt_val_pred = gbt_cv_model.transform(val_df)
gbt_val_auc = evaluator.evaluate(gbt_val_pred)
gbt_val_acc = acc_eval.evaluate(gbt_val_pred)
gbt_val_prec = prec_eval.evaluate(gbt_val_pred)
gbt_val_rec = rec_eval.evaluate(gbt_val_pred)
gbt_val_f1 = f1_eval.evaluate(gbt_val_pred)

print(f"\n--- Validation Metrics ---")
print(f"AUC      : {gbt_val_auc:.4f}")
print(f"Accuracy : {gbt_val_acc:.4f}")
print(f"Precision: {gbt_val_prec:.4f}")
print(f"Recall   : {gbt_val_rec:.4f}")
print(f"F1 Score : {gbt_val_f1:.4f}")

# ============================================
# 8. MODEL COMPARISON
# ============================================

print("\n" + "="*70)
print("TUNED MODELS COMPARISON - VALIDATION SET")
print("="*70)

results = {
    "Logistic Regression": {
        "auc": lr_val_auc, "acc": lr_val_acc, "prec": lr_val_prec, 
        "rec": lr_val_rec, "f1": lr_val_f1, "model": lr_cv_model
    },
    "Random Forest": {
        "auc": rf_val_auc, "acc": rf_val_acc, "prec": rf_val_prec,
        "rec": rf_val_rec, "f1": rf_val_f1, "model": rf_cv_model
    },
    "GBT": {
        "auc": gbt_val_auc, "acc": gbt_val_acc, "prec": gbt_val_prec,
        "rec": gbt_val_rec, "f1": gbt_val_f1, "model": gbt_cv_model
    }
}

print(f"\n{'Model':<25} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 85)
for name, metrics in results.items():
    print(f"{name:<25} {metrics['auc']:<10.4f} {metrics['acc']:<10.4f} "
          f"{metrics['prec']:<10.4f} {metrics['rec']:<10.4f} {metrics['f1']:<10.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model_obj = results[best_model_name]['model']
best_auc = results[best_model_name]['auc']

print(f"\n🏆 Best Model: {best_model_name} (AUC = {best_auc:.4f})")

# ============================================
# 9. EVALUATE ON TEST SET
# ============================================

print("\n" + "="*70)
print("EVALUATING BEST MODEL ON TEST SET")
print("="*70)

test_pred = best_model_obj.transform(test_df)

test_auc = evaluator.evaluate(test_pred)
test_acc = acc_eval.evaluate(test_pred)
test_prec = prec_eval.evaluate(test_pred)
test_rec = rec_eval.evaluate(test_pred)
test_f1 = f1_eval.evaluate(test_pred)

print(f"\nBest Model ({best_model_name}) - Test Set Results:")
print(f"AUC      : {test_auc:.4f}")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1 Score : {test_f1:.4f}")

print("\nConfusion Matrix (Test Set):")
test_pred.groupBy(target_col, "prediction").count().orderBy(target_col, "prediction").show()

# ============================================
# 10. SAVE TUNED MODELS LOCALLY
# ============================================

print("\n" + "="*70)
print("SAVING TUNED MODELS")
print("="*70)

import os

# Create models directory if it doesn't exist
if not os.path.exists("./models"):
    os.makedirs("./models")
    print("Created ./models directory")

# Save all three models
for name, data in results.items():
    model_name = name.replace(" ", "_").lower()
    save_path = f"./models/tuned_{model_name}"
    
    try:
        data['model'].write().overwrite().save(save_path)
        print(f"✓ Saved: {save_path}")
    except Exception as e:
        print(f"✗ Error saving {name}: {str(e)}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Upload tuned models back to GCS:")
print("   gsutil -m cp -r ./models/tuned_* gs://bd-bucket-01/models/")
print("\n2. In Dataproc, load best model and evaluate:")
print(f"   from pyspark.ml import PipelineModel")
print(f"   model = PipelineModel.load('gs://bd-bucket-01/models/tuned_{best_model_name.replace(' ', '_').lower()}')")
print(f"   predictions = model.transform(test_df)")

# Cleanup
train_df.unpersist()
val_df.unpersist()
test_df.unpersist()
spark.stop()

print("\n✅ Hyperparameter tuning complete!")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Test AUC: {test_auc:.4f}")