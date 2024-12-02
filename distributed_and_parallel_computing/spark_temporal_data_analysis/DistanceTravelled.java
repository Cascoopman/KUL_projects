import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;

public class DistanceTravelled {
    public static void main(String[] args) {
        // Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("Distance Travelled")
                .getOrCreate();

        // READ TRAJECTORY DATA:
        // Firstly, add a monotonically increasing index column to keep the order        
        // Secondly, filter out rows representing the ball (player_id = -1)
        // Thirdly, drop irrelevant columns
        Dataset<Row> trajectoryData = spark.read()
                .option("header", true)
                .csv("/data/nba_movement_data/moments/*.csv")
                .withColumn("index", functions.monotonically_increasing_id())
                .filter(col("player_id").notEqual("-1"))
                .drop("team_id", "radius");

        // EUCLIDEAN DISTANCE:
        // Calculate the distance between consecutive moments for each player.
        // Cleanup the data and slim the dataset for efficiency
        Dataset<Row> distanceData = trajectoryData
                .withColumn("x_diff", col("x_loc").minus(lag("x_loc", 1).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index"))))
                .withColumn("y_diff", col("y_loc").minus(lag("y_loc", 1).over(Window.partitionBy("game_id", "event_id", "player_id").orderBy("index"))))
                .withColumn("distance", sqrt(pow(col("x_diff"), 2).plus(pow(col("y_diff"), 2))))
                .drop("x_diff", "y_diff");

        // REMOVE DUPLICATES:
        // Events describe different 'actions' but can have overlapping movements. 
        // Players can only be once in a certain place and time, allowing us to identify duplicates.
        Dataset<Row> duplicatesRemoved = distanceData
                .dropDuplicates("player_id", "x_loc", "y_loc", "quarter", "game_clock", "shot_clock", "game_id")
                .drop("game_clock", "shot_clock", "x_loc", "y_loc", "index");

        // Define a constant for converting feet to meters
        double FEET_TO_METERS = 0.3048;

        // AGGREGATE DISTANCES:
        // Calculate the total travel distance for each player, per game.
        Dataset<Row> totalDistance = duplicatesRemoved
                .groupBy("player_id", "game_id")
                .agg(sum("distance").multiply(FEET_TO_METERS).alias("total_distance_meter"));
        
        // READ MINUTES_PLAYED DATA:
        Dataset<Row> minutesPlayed = spark.read()
                .option("header", true)
                .csv("/data/nba_movement_data/minutes_played.csv");
        
        // JOIN MINUTES_PLAYED WITH TOTAL_DISTANCE:
        // Add the 'seconds played' column to the total distance dataset
        // and divide the distance with seconds to normalize
        // multiply with a quarter (720 seconds) to find the distance per quarter
        // Finally, cast to integer to remove decimal points
        Dataset<Row> distancePerQuarter = totalDistance
                .join(minutesPlayed, totalDistance.col("player_id").equalTo(minutesPlayed.col("PLAYER_ID"))
                .and(totalDistance.col("game_id").equalTo(minutesPlayed.col("GAME_ID"))))
                .withColumn("distance_per_quarter", round(col("total_distance_meter").multiply(720).divide(col("SEC"))))
                .drop(totalDistance.col("player_id"))
                .drop(totalDistance.col("game_id"))
                .drop("SEC", "total_distance_meter");
                  
        // AVERAGE DISTANCE PER QUARTER:
        Dataset<Row> avgDistancePerQuarter = distancePerQuarter
                .groupBy("player_id")
                .agg(avg("distance_per_quarter").cast("int").alias("avg_distance_per_quarter"));
      
        // WRITE TO CSV:
        avgDistancePerQuarter.coalesce(1).write().option("sep", " ").format("csv").mode(SaveMode.Overwrite).save("distance_per_player.csv");
        
        // Stop Spark session
        spark.stop();
    }
}
