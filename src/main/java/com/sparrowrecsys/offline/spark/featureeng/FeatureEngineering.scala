package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {
  /**
   * One-hot encoding example function
   * @param samples movie samples dataframe
   */
  def oneHotEncoderExample(samples:DataFrame): Unit ={
    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))
    //利用Spark的机器学习库Spark MLlib创建One-hot编码器
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      .setDropLast(false)
    //训练One-hot编码器，并完成从id特征到One-hot向量的转换
    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    //打印最终样本的数据结构
    oneHotEncoderSamples.printSchema()
    //打印10条样本查看结果
    oneHotEncoderSamples.show(10)
  }
  // 稀疏向量处理  [4, 0, 3] -》 movieIdVector  (1001,[1],[1.0]) 、 vector (19,[1,2],[1.0,1.0])
  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }

  /**
   * Multi-hot encoding example function
   * @param samples movie samples dataframe
   */
  def multiHotEncoderExample(samples:DataFrame): Unit ={
    val samplesWithGenre = samples.select(col("movieId"), col("title"),explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")

    val stringIndexerModel : StringIndexerModel = genreIndexer.fit(samplesWithGenre)

    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))

    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    val processedSamples =  genreIndexSamples
      .groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize))

    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"),col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10)
  }
  // 稠密向量处理 avgRatingVec [4.165606185002737]
  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

  /**
   * Process rating samples
   * @param samples rating samples
   */
  def ratingFeatures(samples:DataFrame): Unit ={
    samples.printSchema()
    samples.show(10)

    //calculate average movie rating score and rating count
    //利用打分表ratings计算电影的平均分、被打分次数等数值型特征
    val movieFeatures = samples.groupBy(col("movieId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))
      .withColumn("avgRatingVec", double2vec(col("avgRating")))

    movieFeatures.show(10)

    //bucketing
    //分桶处理，创建QuantileDiscretizer进行分桶，将打分次数这一特征分到100个桶中
    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    //Normalization
    val ratingScaler = new MinMaxScaler()
      .setInputCol("avgRatingVec")
      .setOutputCol("scaleAvgRating")

    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("Raw Movie Samples:")
    movieSamples.printSchema()
    movieSamples.show(10)

    println("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)

    println("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)

    println("Numerical features Example:")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingFeatures(ratingSamples)

  }
}
