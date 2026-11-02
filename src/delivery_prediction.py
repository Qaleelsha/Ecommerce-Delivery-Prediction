
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, dayofweek

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# 1. Create local output directory
output_dir = Path("outputs").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

model_path = (output_dir / "delivery_model").as_uri()
predictions_path = (output_dir / "predictions").as_uri()


# 2. Create Spark Session
spark = SparkSession.builder \
    .appName("EcommerceDeliveryPrediction") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# 3. Load Dataset
file_path = Path("dataset/supply_chain.csv").resolve().as_uri()

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("\nOriginal Dataset Rows:", df.count())


# 4. Select Relevant Columns
selected_columns = [
    "order date (DateOrders)",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market",
    "Days for shipment (scheduled)",
    "Late_delivery_risk"
]

data = df.select(selected_columns)


# 5. Data Cleaning
data = data.dropna(subset=selected_columns)

data = data.fillna({
    "Shipping Mode": "Unknown",
    "Order Region": "Unknown",
    "Order Country": "Unknown",
    "Market": "Unknown"
})

data = data.filter(
    col("Late_delivery_risk").isin(0, 1)
)

print("\n========== DATA CLEANING ==========")
print("Cleaned Dataset Rows:", data.count())


# 6. Feature Engineering
data = data.withColumn(
    "Order_Date",
    to_timestamp(
        col("order date (DateOrders)"),
        "M/d/yyyy H:mm"
    )
)

data = data.withColumn(
    "Order_Month",
    month(col("Order_Date"))
)

data = data.withColumn(
    "Order_Weekday",
    dayofweek(col("Order_Date"))
)

data = data.dropna(
    subset=["Order_Month", "Order_Weekday"]
)


# 7. Split Dataset
train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())


# 8. Categorical Feature Encoding
categorical_columns = [
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market"
]

indexer = StringIndexer(
    inputCols=categorical_columns,
    outputCols=[
        "Shipping_Mode_Index",
        "Order_Region_Index",
        "Order_Country_Index",
        "Market_Index"
    ],
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=[
        "Shipping_Mode_Index",
        "Order_Region_Index",
        "Order_Country_Index",
        "Market_Index"
    ],
    outputCols=[
        "Shipping_Mode_Vector",
        "Order_Region_Vector",
        "Order_Country_Vector",
        "Market_Vector"
    ],
    handleInvalid="keep"
)


# 9. Assemble Features
assembler = VectorAssembler(
    inputCols=[
        "Shipping_Mode_Vector",
        "Order_Region_Vector",
        "Order_Country_Vector",
        "Market_Vector",
        "Days for shipment (scheduled)",
        "Order_Month",
        "Order_Weekday"
    ],
    outputCol="features"
)


# 10. Decision Tree Classifier
classifier = DecisionTreeClassifier(
    labelCol="Late_delivery_risk",
    featuresCol="features",
    maxDepth=5,
    seed=42
)


# 11. Create ML Pipeline
pipeline = Pipeline(stages=[
    indexer,
    encoder,
    assembler,
    classifier
])


# 12. Train Model
model = pipeline.fit(train_data)

print("\n========== MODEL TRAINING ==========")
print("Decision Tree Model Trained Successfully!")


# 13. Generate Predictions
predictions = model.transform(test_data)

print("\n========== SAMPLE PREDICTIONS ==========")

predictions.select(
    "Late_delivery_risk",
    "prediction",
    "probability"
).show(10, truncate=False)


# 14. Prediction Distribution
print("\n========== PREDICTION DISTRIBUTION ==========")

predictions.groupBy(
    "prediction"
).count().show()


# 15. Confusion Matrix
print("\n========== CONFUSION MATRIX ==========")

predictions.groupBy(
    "Late_delivery_risk",
    "prediction"
).count().orderBy(
    "Late_delivery_risk",
    "prediction"
).show()


# 16. Model Evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="Late_delivery_risk",
    predictionCol="prediction"
)

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = evaluator.setMetricName("f1").evaluate(predictions)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy, 4))
print("Weighted Precision:", round(precision, 4))
print("Weighted Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))


# 17. Save Trained Model to Local Mac Storage
print("\n========== SAVING MODEL ==========")

model.write().overwrite().save(model_path)

print("Trained Model Saved Successfully!")
print("Model Location:", model_path)


# 18. Export Predictions to Local CSV Folder
print("\n========== EXPORTING PREDICTIONS ==========")

predictions.select(
    "Late_delivery_risk",
    "prediction",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market",
    "Days for shipment (scheduled)",
    "Order_Month",
    "Order_Weekday"
).coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(predictions_path)

print("Predictions Exported Successfully!")
print("Predictions Location:", predictions_path)


# 19. Stop Spark
spark.stop()

print("\n========== PROJECT EXECUTION COMPLETED ==========")
