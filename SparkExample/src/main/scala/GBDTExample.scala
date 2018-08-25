/**
  * Author: songjun
  * Date: 2018/8/23
  * Description: Input:
  * Output:
  * Usage: 
  */
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import scala.collection.mutable.ArrayBuffer

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

    // Prepare test data.
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    val leafNodeIds = getModelLeafNodes(model)

    val testNewFeature = test.
      map{row=>
        val label = row.getAs[Double]("label")
        val fea = row.getAs[Vector]("features")
        val newFea = getGBDTPredictFeatures(leafNodeIds, model, fea)
        (label, fea, newFea)
      }.toDF("label", "fea", "newFea")

    testNewFeature.show(truncate=false)
    /**
      * +------------------------------+
      * |newFea                        |
      * +------------------------------+
      * |[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]|
      * |[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]|
      * |[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]|
      * +------------------------------+
      */

  }

  def getGBDTPredictFeatures(leafNodeIds:Array[String], model:GBTClassificationModel, features:Vector):Array[Double] = {
    var newFeature = new Array[Double](leafNodeIds.length)
    val predNodeIds = getModelPredictNodeIds(model, features)
    for(nodeId <- predNodeIds){
      newFeature(leafNodeIds.indexOf(nodeId)) = 1.0
    }
    newFeature
  }

  def getModelPredictNodeIds(model:GBTClassificationModel, features: Vector) : Array[String] = {
    val predNodeIds = new ArrayBuffer[String]()
    var treeId = 0
    for(tree <- model.trees){
      val uid = treeId.toString
      val nodeId = getTreePredictNodeId(tree.rootNode, uid, features)
      predNodeIds.append(nodeId)
      treeId = treeId + 1
    }
    predNodeIds.toArray
  }

  def getTreePredictNodeId(node: Node, uid: String, features: Vector):String = {
    if(node.isInstanceOf[LeafNode]){
      return uid
    }else{
      val itlNode = node.asInstanceOf[InternalNode]
      if(goToLeft(itlNode, features)){
        return getTreePredictNodeId(itlNode.leftChild, uid+"0", features)
      }else{
        return getTreePredictNodeId(itlNode.rightChild, uid+"1", features)
      }
    }
  }

  def goToLeft(node:InternalNode, features: Vector) : Boolean = {
    val spt = node.split
    spt match {
      case catSpt: CategoricalSplit => return isCategoricalToLeft(catSpt, features)
      case conSpt: ContinuousSplit => return isContinuousToLeft(conSpt, features)
      case _ => throw new Exception("split type not recognize.")
    }
  }

  def isCategoricalToLeft(split: CategoricalSplit, features: Vector):Boolean = {
    split.leftCategories.contains(features(split.featureIndex))
  }

  def isContinuousToLeft(split: ContinuousSplit, features: Vector):Boolean = {
    features(split.featureIndex) <= split.threshold
  }

  def getModelLeafNodes(model:GBTClassificationModel) : Array[String] ={
    val nodeIds = new ArrayBuffer[String]()
    var treeId = 0
    for(tree <- model.trees){
      val uid = treeId.toString
      getTreeLeafNodes(tree.rootNode, uid, nodeIds)
      treeId = treeId + 1
    }
    nodeIds.toArray
  }

  def getTreeLeafNodes(node:Node, uid:String, nodeIds:ArrayBuffer[String]) : Unit = {
    if(node.isInstanceOf[LeafNode]){
      nodeIds.append(uid)
    }else{
      val itlNode = node.asInstanceOf[InternalNode]
      getTreeLeafNodes(itlNode.leftChild, uid+"0", nodeIds)
      getTreeLeafNodes(itlNode.rightChild, uid+"1", nodeIds)
    }
  }


}