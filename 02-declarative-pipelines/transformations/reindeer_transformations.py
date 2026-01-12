from pyspark import pipelines as dp
from pyspark.sql.functions import *
from pyspark.sql.types import *

# --- Configuration ---
# If you changed the below variables in setup.ipynb, then update them here as well
TARGET_CATALOG = "12daysofdemos"
TARGET_SCHEMA = "playground"
TARGET_VOLUME = "data"

volume_path = f"/Volumes/{TARGET_CATALOG}/{TARGET_SCHEMA}/{TARGET_VOLUME}/"

# --- Schema Definition ---
telemetry_schema = StructType([
    StructField("reindeer_id", StringType(), True),
    StructField("reindeer_name", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("heart_rate", IntegerType(), True),
    StructField("altitude_feet", DoubleType(), True),
    StructField("speed_mph", DoubleType(), True),
    StructField("body_temperature", DoubleType(), True),
    StructField("hoof_pressure_fl", IntegerType(), True),
    StructField("hoof_pressure_fr", IntegerType(), True),
    StructField("hoof_pressure_rl", IntegerType(), True),
    StructField("hoof_pressure_rr", IntegerType(), True),
    StructField("energy_efficiency", DoubleType(), True),
    StructField("formation_position", IntegerType(), True),
    StructField("cargo_weight_lbs", IntegerType(), True),
    StructField("wind_speed_mph", IntegerType(), True),
    StructField("visibility", DoubleType(), True),
    StructField("aurora_interference", DoubleType(), True),
    StructField("nose_luminosity", DoubleType(), True),
    StructField("flight_status", StringType(), True),
    StructField("training_type", StringType(), True)
])

# ============================================================================== 
# BRONZE LAYER: Raw Ingestion
# ==============================================================================
@dp.table(name="bronze_telemetry_raw")
def bronze_telemetry_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .schema(telemetry_schema)
        .load(volume_path)
    )

# ============================================================================== 
# SILVER LAYER: Dimension Table
# ==============================================================================
@dp.table(name="dim_reindeer")
def dim_reindeer():
    return (
        spark.readStream.table("bronze_telemetry_raw")
        .select(
            "reindeer_id",
            "reindeer_name",
            "formation_position"
        )
        .distinct()
    )

# ============================================================================== 
# SILVER LAYER: Fact Table with Data Quality Rules
# ==============================================================================
@dp.table(name="fact_telemetry")
@dp.expect_all_or_drop({
    "valid_reindeer_id": "reindeer_id IS NOT NULL",
    "valid_heart_rate": "heart_rate BETWEEN 30 AND 250",
    "valid_speed": "speed_mph >= 0",
    "valid_altitude": "altitude_feet >= 0"
})
def fact_telemetry():
    return (
        spark.readStream.table("bronze_telemetry_raw")
        .withColumn("event_timestamp", to_timestamp(col("timestamp")))
        .withColumn("avg_hoof_pressure", 
            (col("hoof_pressure_fl") + col("hoof_pressure_fr") + 
             col("hoof_pressure_rl") + col("hoof_pressure_rr")) / 4)
        .select(
            "reindeer_id",
            "event_timestamp",
            "training_type",
            "flight_status",
            "heart_rate",
            "altitude_feet",
            "speed_mph",
            "body_temperature",
            "energy_efficiency",
            "nose_luminosity",
            "cargo_weight_lbs",
            "wind_speed_mph",
            "visibility",
            "avg_hoof_pressure"
        )
    )

# ============================================================================== 
# GOLD LAYER: Aggregated Metrics
# ==============================================================================
@dp.table(name="gold_reindeer_performance")
def gold_reindeer_performance():
    fact = spark.read.table("fact_telemetry")
    dim = spark.read.table("dim_reindeer")
    
    return (
        fact.join(dim, "reindeer_id")
        .groupBy("reindeer_name", "training_type")
        .agg(
            count("*").alias("total_readings"),
            avg("speed_mph").alias("avg_speed"),
            avg("heart_rate").alias("avg_heart_rate"),
            max("altitude_feet").alias("max_altitude"),
            avg("energy_efficiency").alias("avg_efficiency")
        )
    )