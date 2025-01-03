from pyspark.sql import SparkSession
from pyspark.sql.functions import col, last, first, array,to_date
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from scipy.interpolate import RBFInterpolator
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Battery-development-project").getOrCreate()

data_path_csv='sample_battery_data.csv'
data = spark.read.csv(data_path_csv, header=True)
df100 = data.withColumn("eventdate", to_date(data.eventdatetime))

df_A_final=df100.withColumnRenamed('eventdatetime', 'created_timestamp')
columns_to_select = [
    'created_timestamp', 'vin', 'eventdate', 'ChargerPlugIn',
    'A_BMS_Temp_Sensor_1', 'A_BMS_Temp_Sensor_2', 'A_BMS_Temp_Sensor_3',
    'A_BMS_Temp_Sensor_4', 'A_BMS_Temp_Sensor_5', 'A_BMS_Temp_Sensor_6',
    'A_Coolant_Temp_In', 'A_Coolant_Temp_Out', 'A_Mux_Temp_Data_Counter',
    'A_Pack_Curr_Value', 'A_Pack_Voltage_Value', 'A_SOC_Value',
    'Unique_Cycle_Type', 'Rank','epoch_time']

df_counter_1 = df_A_final.filter((col("A_Mux_Temp_Data_Counter") == 1) | (col("A_Mux_Temp_Data_Counter").isNull())).select(*columns_to_select)
df_counter_2 = df_A_final.filter(col("A_Mux_Temp_Data_Counter") == 2).select(*columns_to_select)


df_counter_2 = (df_counter_2
    .withColumnRenamed('A_BMS_Temp_Sensor_1', 'A_BMS_Temp_Sensor_7')
    .withColumnRenamed('A_BMS_Temp_Sensor_2', 'A_BMS_Temp_Sensor_8')
    .withColumnRenamed('A_BMS_Temp_Sensor_3', 'A_BMS_Temp_Sensor_9')
    .withColumnRenamed('A_BMS_Temp_Sensor_4', 'A_BMS_Temp_Sensor_10')
    .withColumnRenamed('A_BMS_Temp_Sensor_5', 'A_BMS_Temp_Sensor_11')
    .withColumnRenamed('A_BMS_Temp_Sensor_6', 'A_BMS_Temp_Sensor_12'))

df_counter_1 = df_counter_1.withColumn("A_BMS_Temp_Sensor_7", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_8", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_9", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_10", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_11", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_12", F.lit(None))

df_counter_2 = df_counter_2.withColumn("A_BMS_Temp_Sensor_1", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_2", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_3", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_4", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_5", F.lit(None)) \
                           .withColumn("A_BMS_Temp_Sensor_6", F.lit(None))

df_final = df_counter_1.unionByName(df_counter_2)
df_final_ordered = df_final.orderBy('created_timestamp')

sensor_columns = [
    'A_BMS_Temp_Sensor_1', 'A_BMS_Temp_Sensor_2', 'A_BMS_Temp_Sensor_3',
    'A_BMS_Temp_Sensor_4', 'A_BMS_Temp_Sensor_5', 'A_BMS_Temp_Sensor_6',
    'A_BMS_Temp_Sensor_7', 'A_BMS_Temp_Sensor_8', 'A_BMS_Temp_Sensor_9',
    'A_BMS_Temp_Sensor_10', 'A_BMS_Temp_Sensor_11', 'A_BMS_Temp_Sensor_12']

# Define windows for forward and backward interpolation
window_forward = (
    Window.partitionBy("Unique_Cycle_Type").orderBy("created_timestamp")
          .rowsBetween(Window.unboundedPreceding, Window.currentRow))

window_backward = (
    Window.partitionBy("Unique_Cycle_Type").orderBy(F.col("created_timestamp").desc())
          .rowsBetween(Window.unboundedPreceding, Window.currentRow))

# Function to perform forward and backward interpolation without overwriting
def interpolate_grouped_by_vin(df):
    for col in sensor_columns:
        forward_col = col + "_forward"
        df = df.withColumn(forward_col, F.last(col, ignorenulls=True).over(window_forward))
        backward_col = col + "_backward"
        df = df.withColumn(backward_col, F.first(col, ignorenulls=True).over(window_backward))
        df = df.withColumn(
            col,
            F.coalesce(F.col(forward_col), F.col(backward_col)))
        df = df.drop(forward_col, backward_col)
    return df

# Apply interpolation to the entire DataFrame, grouped by Unique_Cycle_Type
df_final_interpolated = interpolate_grouped_by_vin(df_final_ordered)

df_final_interpolated = df_final_interpolated.orderBy("created_timestamp")

df = df_final_interpolated

class BatteryConfig:
    def __init__(self):
        # Locations of sensors for module interpolation
        self.module_sensor_locations = np.array(
            [[50, 113.5], [1300, 113.5], [682, 313.5],
             [682, 515.5], [50, 713.5], [1300, 713.5]])
        # Limits of the battery pack for creating a grid
        self.module_limits = {'bottom': 63.5, 'top': 763.5, 'left': 0, 'right': 1350}


# Instantiate the configuration
battery_config = BatteryConfig()

class Interpolator:
    def __init__(self):
        self.centers = battery_config.module_sensor_locations
        self.pack_limits = battery_config.module_limits
        resolution = 104
        grid = np.mgrid[
            self.pack_limits['left']:self.pack_limits['right']:complex(0, resolution),
            self.pack_limits['bottom']:self.pack_limits['top']:complex(0, resolution)]
        self.point_list = grid.reshape(2, -1).T
    def evaluate(self, values, smooth_factor=0.1):
        rbf = RBFInterpolator(self.centers, values, smoothing=smooth_factor)
        return rbf(self.point_list).flatten()

interpolator = Interpolator()

# Broadcast necessary data
broadcast_grid = spark.sparkContext.broadcast(interpolator.point_list)
broadcast_centers = spark.sparkContext.broadcast(interpolator.centers)

@udf(ArrayType(FloatType()))
def interpolate_udf(values):
    try:
        grid = broadcast_grid.value 
        centers = broadcast_centers.value 
        rbf = RBFInterpolator(centers, values)
        interpolated_values = rbf(grid).tolist()
        return interpolated_values
    except Exception as e:
        raise ValueError(f"Error during interpolation: {e}")

df = df.withColumn(
    "Sensor_1_to_6",
    array(
        col("A_BMS_Temp_Sensor_1").cast("float"),
        col("A_BMS_Temp_Sensor_2").cast("float"),
        col("A_BMS_Temp_Sensor_3").cast("float"),
        col("A_BMS_Temp_Sensor_4").cast("float"),
        col("A_BMS_Temp_Sensor_5").cast("float"),
        col("A_BMS_Temp_Sensor_6").cast("float"),))

df = df.withColumn(
    "Sensor_7_to_12",
    array(
        col("A_BMS_Temp_Sensor_7").cast("float"),
        col("A_BMS_Temp_Sensor_8").cast("float"),
        col("A_BMS_Temp_Sensor_9").cast("float"),
        col("A_BMS_Temp_Sensor_10").cast("float"),
        col("A_BMS_Temp_Sensor_11").cast("float"),
        col("A_BMS_Temp_Sensor_12").cast("float"),))

df_with_interpolated = (
    df.withColumn("Interpolated_1_to_104", interpolate_udf(col("Sensor_1_to_6")))
      .withColumn("Interpolated_105_to_208", interpolate_udf(col("Sensor_7_to_12"))))

# Explode interpolated arrays into individual columns
for i in range(1, 105):
    df_with_interpolated = df_with_interpolated.withColumn(
        f"Sensor_1_to_6_Cell_{i}",
        col("Interpolated_1_to_104")[i - 1])

for i in range(105, 209):
    df_with_interpolated = df_with_interpolated.withColumn(
        f"Sensor_7_to_12_Cell_{i - 104}",
        col("Interpolated_105_to_208")[i - 105])

df_final = df_with_interpolated.drop(
    "Interpolated_1_to_104", "Interpolated_105_to_208", "Sensor_1_to_6", "Sensor_7_to_12")

pandas_Pack_updated = df_final.toPandas()
pandas_Pack_updated.to_csv('/home/hadoop/battery_interpolation.csv', header=True)
