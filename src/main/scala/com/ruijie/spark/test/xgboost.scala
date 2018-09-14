package com.ruijie.spark.test

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * @ author：80374784
  * @ description：
  * @ date：create in 17:09 2018/4/16
  */
object xgboost {
  case class PowerPlantTable(AT: Int, V : Int, AP : Int, RH : Int, PE : Int)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Spark SQL Example")
      .master("local[2]")
      .config("spark.some.config.option", "some-value")
//      .config("spark.sql.warehouse.dir", "file:///e:/tmp/spark-warehouse")
      .getOrCreate()

    System.setProperty("hadoop.home.dir", "D:\\hadooplocal\\hadoop-common-2.2.0-bin-master")
    import spark.implicits._

    spark.sql("drop table if exists power_plant")

    val dataset = spark.sparkContext
      .textFile("E:\\IDEAspace\\sparktest\\src\\resource\\data\\test.csv")
      .map(x => x.split(","))
      .filter(line => line(0) != "AT")
      .map(line => PowerPlantTable(line(0).toInt, line(1).toInt, line(2).toInt, line(3).toInt, line(4).toInt))
      .toDF

    dataset.show()
    val assembler =  new VectorAssembler().setInputCols(Array("AT", "V", "AP", "RH")).setOutputCol("features")
    val vected = assembler.transform(dataset).withColumnRenamed("PE","label").drop("AT","V","AP","RH")
    val Array(split20, split80) = vected.randomSplit(Array(0.20, 0.80), 2)
    val testSet = split20.cache()
    val trainingSet = split80.cache()
    val paramMap = List(
      "eta" -> 0.3,
      "max_depth" -> 6,
      "objective" -> "reg:linear",
      "early_stopping_rounds" ->10).toMap
    val xgboostModel = XGBoost.trainWithDataFrame(trainingSet, paramMap, 30, 2, useExternalMemory=false)
    val predictions = xgboostModel.transform(testSet)   // 这里的predictions是一个dataframe
//    testSet.show()
    predictions.show()
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)     // rmse是一种用于评估回归类型模型的方法
    print ("Root mean squared error: " + rmse)
  }
}
