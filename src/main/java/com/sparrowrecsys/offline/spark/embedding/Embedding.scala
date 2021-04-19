package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.mutable
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object Embedding {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  def processItemSequence(sparkSession: SparkSession, rawSampleDataPath: String): RDD[Seq[String]] ={
    //设定rating数据的路径并用spark载入数据
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    //实现一个用户定义的操作函数(UDF)，用于之后的排序
    val sortUdf: UserDefinedFunction = udf((rows :Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp)}
        .sortBy{ case (_, timestamp) => timestamp}
        .map{case (movieId, _) => movieId}
    })

    ratingSamples.printSchema()

    //把原始的rating数据处理成序列数据
    val userSeq = ratingSamples
      .where(col("rating") >= 3.5)
      .groupBy("userId")
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
      .withColumn("movieIdStr", array_join(col("movieIds"), " "))

    //把序列数据筛选出来，丢掉其他过程数据
    userSeq.select("userId", "movieIdStr").show(10, truncate = false)
    userSeq.select("movieIdStr").rdd.map(r => (r.getAs[String]("movieIdStr").split(" ").toSeq))
  }

  def embeddingLSH(spark:SparkSession, movieEmbMap:Map[String, Array[Float]]): Unit ={
    val movieEmbSeq = movieEmbMap.toSeq.map(item => (item._1, Vectors.dense(item._2.map(f => f.toDouble))))
    val movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")

    // LSH bucket model
    val bucketProjectionLSH = new BucketedRandomProjectionLSH()
      .setBucketLength(0.1)
      .setNumHashTables(3)
      .setInputCol("emb")
      .setOutputCol("bucketId")

    val bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    val embBucketResult = bucketModel.transform(movieEmbDF)
    println("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    println("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate = false)

    println("Approximately searching for 5 nearest neighbors of the sample embedding:")
    val sampleEmb = Vectors.dense(0.795, 0.583,1.120,0.850,0.174,-0.839,-0.0633,0.249,0.673,-0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate = false)
  }

  def trainItem2vec(sparkSession: SparkSession, samples: RDD[Seq[String]], embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel = {
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)   // embedding向量维度
      .setWindowSize(5)           // 序列数据滑动窗口大小
      .setNumIterations(10)       // 训练迭代次数

    // 训练模型，返回一个包含所有模型参数的对象
    val model = word2vec.fit(samples)

    val synonyms = model.findSynonyms("158", 20)
    for((synonyms, cosineSimilarity) <- synonyms) {
      println(s"$synonyms $cosineSimilarity")
    }

    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    for(movieId <- model.getVectors.keys) {
      bw.write(movieId + ":" + floatArrayOps(model.getVectors(movieId)).mkString(" ") + "\n")
    }
    bw.close()

    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      params.ex(60*60*24)
      // key的形式为前缀+movieId，例如i2vEmb:361
      // value的形式是由Embedding向量生成的字符串，例如 "0.1693846 0.2964318 -0.13044095 0.37574086 0.55175656 0.03217995 1.327348 -0.81346786 0.45146862 0.49406642"
      for (movieId <- model.getVectors.keys) {
        redisClient.set(redisKeyPrefix + ":" + movieId, floatArrayOps(model.getVectors(movieId)).mkString(" "), params)
      }
      redisClient.close()
    }

    embeddingLSH(sparkSession, model.getVectors)
    model
  }

  //samples 输入的观影序列样本集
  def generateTransitionMatrix(samples: RDD[Seq[String]]): (mutable.Map[String, mutable.Map[String, Double]], mutable.Map[String, Double]) ={
    // 通过flatMap操作把观影序列打碎成一个个影片对
    val pairSamples = samples.flatMap[(String, String)](sample => {
      var pairSeq = Seq[(String, String)]()
      var previousItem:String = null
      sample.foreach((element:String) => {
        if(previousItem != null) {
          pairSeq = pairSeq :+ (previousItem, element)
        }
        previousItem = element
      })
      pairSeq
    })
    //统计影片对的数量
    val pairCountMap = pairSamples.countByValue()
    var pairTotalCount = 0L
    //转移概率矩阵的双层Map数据结构
    val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
    val itemCountMap = mutable.Map[String, Long]()

    //求取转移概率矩阵
    pairCountMap.foreach( pair => {
      val pairItems = pair._1
      val count = pair._2

      if(!transitionCountMatrix.contains(pairItems._1)) {
        transitionCountMatrix(pairItems._1) = mutable.Map[String, Long]()
      }
      transitionCountMatrix(pairItems._1)(pairItems._2) = count
      itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1, 0) + count
      pairTotalCount = pairTotalCount + count
    })

    val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
    val itemDistribution = mutable.Map[String, Double]()

    transitionCountMatrix foreach {
      case (itemAId, transitionMap) =>
        transitionMatrix(itemAId) = mutable.Map[String, Double]()
        transitionMap foreach {case (itemBId, transitionCount) => transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble / itemCountMap(itemAId)}
    }
    itemCountMap foreach {case (itemId, itemCount) => itemDistribution(itemId) = itemCount.toDouble / pairTotalCount}
    (transitionMatrix, itemDistribution)

  }

  //随机游走采样函数
  // transferMatrix 转移概率矩阵
  // itemDistribution 物品出现的分布
  def randomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleCount: Int, sampleLength: Int): Seq[Seq[String]] ={
    val samples = mutable.ListBuffer[Seq[String]]()
    //随机游走sampleCount次，生成sampleCount个序列样本
    for(_ <- 1 to sampleCount) {
      samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    }
    Seq(samples.toList : _*)
  }

  /**
   * 通过随机游走生成样本的过程
   * @param transitionMatrix  转移概率矩阵
   * @param itemDistribution  物品出现的分布
   * @param sampleLength  每个样本的长度
   * @return
   */
  def oneRandomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleLength: Int): Seq[String] = {
    val sample = mutable.ListBuffer[String]()

    // pick the first element
    val randomDouble = Random.nextDouble()
    var firstItem = ""
    var accumulateProb:Double = 0D
    breakable { for ((item, prob) <- itemDistribution) {
      accumulateProb += prob
      if (accumulateProb >= randomDouble) {
        firstItem = item
        break
      }
    }}

    sample.append(firstItem)
    var curElement = firstItem

    breakable { for(_ <- 1 until sampleLength) {
      if (!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)) {
        break
      }

      val probDistribution = transitionMatrix(curElement)
      val randomDouble = Random.nextDouble()
      breakable { for ((item, prob) <- probDistribution) {
        if (randomDouble >= prob) {
          curElement = item
          break
        }
      }}
      sample.append(curElement)
    }}
    Seq(sample.toList : _*)
  }

  def graphEmb(samples : RDD[Seq[String]], sparkSession: SparkSession, embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel ={
    val transitionMatrixAndItemDis = generateTransitionMatrix(samples)

    println(transitionMatrixAndItemDis._1.size)
    println(transitionMatrixAndItemDis._2.size)
    //样本的数量
    val sampleCount = 20000
    //每个样本的长度
    val sampleLength = 10
    val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)

    val rddSamples = sparkSession.sparkContext.parallelize(newSamples)
    trainItem2vec(sparkSession, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)
  }


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10

    val samples = processItemSequence(spark, rawSampleDataPath)
    val model = trainItem2vec(spark, samples, embLength, "item2vecEmb.csv", saveToRedis = false, "i2vEmb")

  }
}
