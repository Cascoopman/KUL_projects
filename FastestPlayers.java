import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;

public class FastestPlayers {
    public static void main(String[] args) {
        // Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("Fastest Players")
                .getOrCreate();

        // READ TRAJECTORY DATA:
        // Add a monotonically increasing index column to keep the order        
        // Filter out rows representing the ball (player_id = -1)
        // And explicitly drop irrelevant columns
        Dataset<Row> trajectoryData = spark.read()
                .option("header", true)
                .csv("/data/nba_movement_data/moments/*.csv")
                .withColumn("index", functions.monotonically_increasing_id())
                .filter(col("player_id").notEqual("-1"))
                .drop("team_id", "radius", "shot_clock", "game_clock", "quarter");

        // Define a constant for converting feet to meters
        double FEET_TO_METERS = 0.3048;   
            
        // Define the measurement interval constant
        double DELTA = 1/25.0;

        // SPEED CALCULATION:
        // Calculate the speed for each player.
        // Removing the speed outliers (above 12 m/s), and
        // smoothing the speed values over a window of 10 rows
        Dataset<Row> speedData = trajectoryData
                .withColumn("x_diff", col("x_loc").minus(lag("x_loc", 1).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index"))))
                .withColumn("y_diff", col("y_loc").minus(lag("y_loc", 1).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index"))))
                .withColumn("sqrt_sum", sqrt(pow(col("x_diff"), 2).plus(pow(col("y_diff"), 2))))
                .withColumn("speed", col("sqrt_sum").multiply(FEET_TO_METERS).divide(DELTA))
                .filter(col("speed").leq(12.0))
                .withColumn("smoothed_speed", avg(col("speed")).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index").rowsBetween(0, 9)))
                .drop("x_loc", "y_loc", "x_diff", "y_diff", "sqrt_sum", "speed");

        // ACCELERATION CALCULATION:
        // Using the smoothed and filtered speed values to calculate the accelerations
        // Also smoothing the acceleration values over a window of 10 rows
        Dataset<Row> accelerationData = speedData 
                .withColumn("acceleration", col("smoothed_speed").minus(lag("smoothed_speed", 1).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index")))
                        .divide(DELTA))
                .withColumn("smoothed_acc", avg(col("acceleration")).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index").rowsBetween(0, 9)))
                .drop("acceleration")
                .orderBy(col("smoothed_acc").desc());

        // CALCULATE MAX SPEED AND ACCELERATION:
        // Return the highest speed and acceleration values for each player.
        // Round the values to two decimal places and return as space separated csv
        Dataset<Row> fastestPlayers = accelerationData
                .groupBy("player_id").agg(round(max("smoothed_speed"), 2).as("highest_speed"), round(max("smoothed_acc"), 2).as("highest_acc"))
                .drop("game_id", "event_id", "index");

        fastestPlayers.coalesce(1).write().option("sep", " ").format("csv").mode(SaveMode.Overwrite).save("fastest_players.csv");

        // Stop Spark session
        spark.stop();
    }
}