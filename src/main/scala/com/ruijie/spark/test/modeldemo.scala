package com.ruijie.spark.test

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, udf}

object modeldemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("titanic-pipline")
      .getOrCreate()
    System.setProperty("hadoop.home.dir", "D:\\hadooplocal\\hadoop-common-2.2.0-bin-master")
    var raw = spark
      .read
      .format("csv")
      .option("header",true)
      .option("inferSchema",true.toString)
      .load("E:\\IDEAspace\\sparktest\\src\\resource\\data\\train.csv")

    raw.show(3)
    import spark.sqlContext.implicits._
    val getFirst = udf((inputStr:String)=>{
      inputStr match{
        case inputStr:String =>inputStr.substring(0,1)
        case _=>null
      }

    })

    //处理cabin
    raw = raw.withColumn("CabinFirst",getFirst($"Cabin"))
    raw = raw.na.fill("O",Array("CabinFirst"))

    val cabinStringIndexer = new StringIndexer()
      .setInputCol("CabinFirst")
      .setOutputCol("CabinIndexer")

    val cabinHotEncoder = new OneHotEncoder()
      .setInputCol(cabinStringIndexer.getOutputCol)
      .setOutputCol("CabinVector")
      .setDropLast(false)

    //处理embarked
    val emarkedRaw = raw.groupBy("Embarked").count()
    val emarkedRawMost = emarkedRaw.orderBy(-emarkedRaw("count")).head()(0)
    raw  = raw.na.fill(emarkedRawMost.asInstanceOf[String],Array("Embarked"))

    val EmbarkedStringIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndexer")

    val EmbarkedOneHotEncoder = new OneHotEncoder()
      .setInputCol(EmbarkedStringIndexer.getOutputCol)
      .setOutputCol("EmbarkedVector")
      .setDropLast(false)

    //处理年龄
    val avgAge = raw.agg(avg("Age")).head()
    raw = raw.na.fill(avgAge(0).asInstanceOf[Double],Array("Age"))

    //处理性别
    val sexStringIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndexer")

    //获取需要的特征字段
    val featuresColum = Array("Pclass","Age"
      ,"SibSp","Parch","Fare"
      ,"CabinVector","EmbarkedVector","SexIndexer")

    val titanicAssemble = new VectorAssembler()
      .setInputCols(featuresColum)
      .setOutputCol("features")

    //获取逻辑回归训练模型
    val lr= new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("Survived")
      .setMaxIter(30)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)


    //ml训练管道
    val pipelineStage = Array(cabinStringIndexer,cabinHotEncoder,EmbarkedStringIndexer
      ,EmbarkedOneHotEncoder,sexStringIndexer,titanicAssemble,lr)
    val pipline = new Pipeline().setStages(pipelineStage)

    //获取训练集和测试集
    //划分数据集
    val rawAll = raw.randomSplitAsList(Array(0.2,0.8),10)

    val train = rawAll.get(1)
    val test = rawAll.get(0)

    //获取模型
    val pModle = pipline.fit(train)


    val testPrediction = pModle.transform(test).select("prediction","Survived")

    //得到准确率
    val correctCount = testPrediction.filter(testPrediction("prediction").equalTo(testPrediction("Survived"))).count()

    val totalCount = testPrediction.count()

    println("正确的数量为： " + correctCount + "\n"
      +"总数量为： " + totalCount + "\n"
      +"准确率为： " + correctCount/totalCount.asInstanceOf[Double])

  }
}
