/**
  * Author: songjun
  * Date: 2018/12/28
  */

import org.apache.commons.cli.Options
import org.apache.commons.cli.PosixParser
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.LogisticRegression

object LRExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().
      getOrCreate()
    import spark.implicits._

    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val lr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.3).
      setElasticNetParam(0.8)

    val lrModel = lr.fit(training)

    println(s"Coefficients ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")

    val mlr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.3).
      setElasticNetParam(0.8).
      setFamily("multinomial")

    val mlrModel = mlr.fit(training)

    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

  }
}
