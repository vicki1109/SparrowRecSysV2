//package com.sparrowrecsys.offline.spark.featureeng
//
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.SparkConf
//import org.apache.spark.sql.expressions.UserDefinedFunction
//import org.apache.spark.sql.{DataFrame, SparkSession}
//import org.apache.spark.sql.functions.{format_number, _}
//
//import scala.collection.immutable.ListMap
//import scala.collection.{JavaConversions, mutable}
//
//
//object FeatureEngForRecModel {
//  val NUMBER_PRECISION = 2
//  val redisEndpoint = "localhost"
//  val redisPort = 6379
//
//  def addSampleLable(ratingSamples:DataFrame): DataFrame ={
//    ratingSamples.show(10, truncate = false)
//    ratingSamples.printSchema()
//    val sampleCount = ratingSamples.count()
//    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
//      .withColumn("percentage", col("count")/sampleCount).show(100, truncate = false)
//    ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
//  }
//
//  def addMovieFeatures(movieSamples:DataFrame, ratingSamples:DataFrame): DataFrame ={
//    // add movie basic features
//    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
//    // add release year
//    val extractReleaseYearUdf = udf({(title: String) => {
//      if(null == title || title.trim.length < 6) {
//        1990
//      }
//      else {
//        val yearString = title.trim.substring(title.length - 5, title.length - 1)
//        yearString.toInt
//      }
//    }})
//
//    // add title
//    val extractTitleUdf = udf({(title: String) => {title.trim.substring(0, title.trim.length - 6).trim}})
//
//    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
//      .withColumn("title", extractTitleUdf(col("title")))
//      .drop("title")  // title is useless currently
//
//    // split genres
//    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1", split(col("genres"), "\\|").getItem(0))
//      .withColumn("movieGenre2", split(col("genres"), "\\|").getItem(1))
//      .withColumn("movieGenre3", split(col("genres"), "\\|").getItem(2))
//
//    // add rating features
//    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
//      .agg(count(lit(1)).as("movieRatingCount"),
//        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
//        stddev(col("rating")).as("movieRatingStddev"))
//      .na.fill(0).withColumn("movieRatingStddev", format_number(col("movieRatingStddev"), NUMBER_PRECISION))
//
//    // join movie rating features
//    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
//    samplesWithMovies4.printSchema()
//    samplesWithMovies4.show(10, truncate=false)
//
//    samplesWithMovies4
//  }
//
//  val extractGenres:UserDefinedFunction = udf{ (genreArray: Seq[String]) => {
//    val genreMap = mutable.Map[String, Int]()
//    genreArray.foreach((element:String) => {
//      val genres = element.split("\\|")
//      genres.foreach((oneGenre:String) => {
//        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0) + 1
//      })
//    })
//    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2):_*)
//    sortedGenres.keys.toSeq
//  } }
//
//  def addUserFeatures(ratingSamples:DataFrame): DataFrame ={
//    val samplesWithUserFeatures = ratingSamples
//      .withColumn("")
//  }
//
//  def main(args: Array[String]): Unit = {
//    Logger.getLogger("org").setLevel(Level.ERROR)
//
//    val conf = new SparkConf()
//      .setMaster("local")
//      .setAppName("featureEngineering")
//      .set("spark.submit.deployMode", "client")
//
//    val spark = SparkSession.builder.config(conf).getOrCreate()
//    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
//    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
//
//    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
//    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
//
//    val ratingSamplesWithLabel = addSampleLable(ratingSamples)
//    ratingSamplesWithLabel.show(10, truncate = false)
//
//    val samplesWithMovieFeatures = addMovieFeatures()
//  }
//}
