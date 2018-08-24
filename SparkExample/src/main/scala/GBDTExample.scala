/**
  * Author: songjun
  * Date: 2018/8/23
  * Description: Input:
  * Output:
  * Usage: 
  */
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree.{InternalNode, LeafNode}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
object GBDTExample{
  def main(args:Array[String]):Unit={
    val spark = SparkSession.builder().
      getOrCreate()
    import spark.implicits._

    // Prepare training data from a list of (label, features) tuples.
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    val numTrees = 3
    val gbt = new GBTClassifier().setMaxIter(3).setMaxDepth(4)

    val model = gbt.fit(training)

    val tree0 = model.trees(0)
    println(s"tree 0 nodes num: ${tree0.numNodes}")


  }
}