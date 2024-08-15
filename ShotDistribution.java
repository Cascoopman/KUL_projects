import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.Window;

public class ShotDistribution {

    public static void main(String[] args) {
        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("ShotDistribution")
                .getOrCreate();
        
        // Load trajectory data of both Stephen Curry and the ball
        Dataset<Row> curryAndBallMovementData = spark.read()
                .option("header", true)
                .csv("/data/nba_movement_data/moments/*.csv")
                .select("player_id", "game_id", "event_id", "x_loc", "y_loc", "radius", "game_clock", "shot_clock")
                .filter(col("player_id").equalTo(-1).or(col("player_id").equalTo(201939)));

        // Load play-by-play (event) data and
        // filter event for Stephen Curry's shots
        Dataset<Row> curryShotEvents = spark.read()
                .option("header", true)
                .csv("/data/nba_movement_data/events/*.csv")
                .filter(col("PLAYER1_ID").equalTo(201939))
                .filter(col("EVENTMSGTYPE").equalTo(1).or(col("EVENTMSGTYPE").equalTo(2)))
                .select("GAME_ID","EVENTNUM", "EVENTMSGTYPE")
                .withColumnRenamed("GAME_ID", "game_id_two");

        // Join event data with trajectory data
        Dataset<Row> joinedData = curryShotEvents.join(curryAndBallMovementData,
                curryShotEvents.col("game_id_two").equalTo(curryAndBallMovementData.col("game_id"))
                        .and(curryShotEvents.col("EVENTNUM").equalTo(curryAndBallMovementData.col("event_id"))))
                .drop("game_id_two", "EVENTNUM")
                .withColumn("index", functions.monotonically_increasing_id());

        // ANALYSE RELATIVE BALL MOVEMENT (-1);
        // Its difference to Curry's position and
        // its distance to the 2 hoops
        Dataset<Row> ballRelativeMovementData = joinedData
                .withColumn("x_distance_to_curry", col("x_loc").minus(lead("x_loc", 1).over(Window.partitionBy("game_id", "event_id", "game_clock", "shot_clock").orderBy("index"))))
                .withColumn("y_distance_to_curry", col("y_loc").minus(lead("y_loc", 1).over(Window.partitionBy("game_id", "event_id", "game_clock", "shot_clock").orderBy("index"))))
                .withColumn("distance_to_curry", sqrt(pow(col("x_distance_to_curry"), 2).plus(pow(col("y_distance_to_curry"), 2))))
                .withColumn("distance_to_hoop_1", sqrt(pow(col("x_loc").minus(5), 2).plus(pow(col("y_loc").minus(25), 2))))
                .withColumn("distance_to_hoop_2", sqrt(pow(col("x_loc").minus(89), 2).plus(pow(col("y_loc").minus(25), 2))))        
                .drop("x_distance_to_curry", "y_distance_to_curry")
                .withColumnRenamed("radius", "ball_height");
        
        // IDENTIFYING SHOTS: ENDING OF SHOT ARC
        // Find all the points where the ball is higher than 10 feet. 
        // (official rim height)
        // Identify the first point, per event, near a rim (2 feet of either rim)
        // (Assuming even missed shots are close to the rim)
        // We see this as the end of the shot arc
        Dataset<Row> endingShotArc = ballRelativeMovementData
                .filter(col("ball_height").geq(10))
                .filter(col("distance_to_hoop_1").leq(2).or(col("distance_to_hoop_2").leq(2)))
                .groupBy("game_id", "event_id")
                .agg(min("index").as("end_of_shot_arc"));

        // IDENTIFYING SHOTS: BEGINNING OF SHOT ARC
        // Find all the points before that where Curry had the ball
        Dataset<Row> beforeShotArc = ballRelativeMovementData.join(
                endingShotArc,
                ballRelativeMovementData.col("game_id").equalTo(endingShotArc.col("game_id"))
                        .and(ballRelativeMovementData.col("event_id").equalTo(endingShotArc.col("event_id")))
                        .and(ballRelativeMovementData.col("index").leq(endingShotArc.col("end_of_shot_arc"))),
                "inner"
                )
                .filter(ballRelativeMovementData.col("distance_to_curry").leq(1))
                .drop(endingShotArc.col("game_id")).drop(endingShotArc.col("event_id"));

        // Find the last point where he had possession, 
        // indicating the location where he let go
        Dataset<Row> locationShotBeginning = beforeShotArc
                .groupBy("game_id", "event_id")
                .agg(
                        max("index").as("moment_of_release")
                )
                .withColumnRenamed("game_id", "game_id_two")
                .withColumnRenamed("event_id", "event_id_two");

        // Join the data to get the exact location of the shot beginning
        // Less efficient to redo join operation but more precise
        // than alternatives
        locationShotBeginning = locationShotBeginning.join(
                ballRelativeMovementData,
                locationShotBeginning.col("game_id_two").equalTo(ballRelativeMovementData.col("game_id"))
                        .and(locationShotBeginning.col("event_id_two").equalTo(ballRelativeMovementData.col("event_id")))
                        .and(locationShotBeginning.col("moment_of_release").equalTo(ballRelativeMovementData.col("index"))),
                "inner"
                )
                .withColumnRenamed("x_loc", "x_loc_shot")
                .withColumnRenamed("y_loc", "y_loc_shot")
                .drop(ballRelativeMovementData.col("game_id"))
                .drop(ballRelativeMovementData.col("event_id"))
                .drop(ballRelativeMovementData.col("index"));

        // Transform the data from full court to half court
        // If curry stands accros halfway, flip the coordinates over the center point
        Dataset<Row> curryLocationShotBeginningTransformed = locationShotBeginning
                .withColumn("x_loc_new", when(col("x_loc_shot").geq(47), lit(94).minus(col("x_loc_shot"))).otherwise(col("x_loc_shot")))
                .withColumn("y_loc_new", when(col("x_loc_shot").geq(47), lit(50).minus(col("y_loc_shot"))).otherwise(col("y_loc_shot")))
                .withColumn("bin_x", round(col("x_loc_new"), 0))
                .withColumn("bin_y", round(col("y_loc_new"), 0))
                .select("bin_x", "bin_y", "EVENTMSGTYPE");

        // Output the data to a csv file
        Dataset<Row> shotsMadeCurryPerLocation = curryLocationShotBeginningTransformed
                .groupBy("bin_x", "bin_y")
                .agg(
                        sum(when(col("EVENTMSGTYPE").equalTo(2), 1).otherwise(0)).as("shots_missed"),
                        sum(when(col("EVENTMSGTYPE").equalTo(1), 1).otherwise(0)).as("shots_made")
                    )
                .orderBy("bin_x", "bin_y");

        shotsMadeCurryPerLocation.coalesce(1).write().option("sep", " ").option("header", true).format("csv").mode(SaveMode.Overwrite).save("shot_distribution.csv");
        
        // Heatmap is created in python using seaborn

        // Stop SparkSession
        spark.stop();
    }
}
