/**
  * Author: songjun
  * Date: 2018/7/10
  * Description: Input:
  * Output:
  * Usage:
  */
package spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.classification.LogisticRegression
object LRModel{
  def main(args:Array[String]):Unit = {
    val lr = new LogisticRegression().
      setMaxIter(100).
      setRegParam(0.3).
      setElasticNetParam(0.8)

    lr.getProbabilityCol
  }
}