// Modelo de Classificação de Palavras Usando Word2Vec


// Pacotes
package org.apache.spark.mllib.linalg

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.Word2Vec
import scala.util.Try

case class Sample(id: String, review: String, sentiment: Option[Int] = None)

object ModeloWord2Vec extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def printRDD(xs: RDD[_]) {
    println("--------------------------")
    xs take 5 foreach println
    println("--------------------------")
  }

  val conf = new SparkConf(false).setMaster("local").setAppName("Word2Vec")
  val sc = new SparkContext(conf)

  // Carregando Dados de Treino e de Teste
  val trainPath = s"data/labeledTrainData.tsv"
  val testPath = s"data/testData.tsv"

  // Removendo o cabeçalho
  def skipHeaders(idx: Int, iter: Iterator[String]) = if (idx == 0) iter.drop(1) else iter

  // Objetos para dados de treino e de teste
  val trainFile = sc.textFile(trainPath) mapPartitionsWithIndex skipHeaders map (l => l.split("\t"))
  val testFile = sc.textFile(testPath) mapPartitionsWithIndex skipHeaders map (l => l.split("\t"))

  // Função para coletar uma amostra (sample)
  def toSample(segments: Array[String]) = segments match {
    case Array(id, sentiment, review) => Sample(id, review, Some(sentiment.toInt))
    case Array(id, review) => Sample(id, review)
  }

  // Coletando amostras
  val trainSamples = trainFile map toSample
  val testSamples = testFile map toSample

  // Funções para limpeza de código Html
  def cleanHtml(str: String) = str.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", "")
  def cleanSampleHtml(sample: Sample) = sample copy (review = cleanHtml(sample.review))

  // Limpeza
  val cleanTrainSamples = trainSamples map cleanSampleHtml
  val cleanTestSamples = testSamples map cleanSampleHtml

  // Função para Tokenização
  def cleanWord(str: String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W", "")).reduce((x, y) => s"$x $y")

  // Função para aplicar tokenização nas amostras de dados
  def wordOnlySample(sample: Sample) = sample copy (review = cleanWord(sample.review))

  // Aplicando tokenização
  val wordOnlyTrainSample = cleanTrainSamples map wordOnlySample
  val wordOnlyTestSample = cleanTestSamples map wordOnlySample

  // Treinando modelo Word2Vec
  val samplePairs = wordOnlyTrainSample.map(s => s.id -> s).cache()
  val reviewWordsPairs: RDD[(String, Iterable[String])] = samplePairs.mapValues(_.review.split(" ").toIterable)

  println("\n<--- Iniciando Treinamento Word2Vec --->\n")
  val word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)

  println("\n<--- Treinamento Finalizado --->\n")
  println(word2vecModel.transform("london"))
  println(word2vecModel.findSynonyms("london", 4))

  // Funções para coletar os pares de features
  def wordFeatures(words: Iterable[String]): Iterable[Vector] = words.map(w => Try(word2vecModel.transform(w))).filter(_.isSuccess).map(_.get)

  def avgWordFeatures(wordFeatures: Iterable[Vector]): Vector = Vectors.fromBreeze(wordFeatures.map(_.toBreeze).reduceLeft(_ + _) / wordFeatures.size.toDouble)

  // Criando feature vectors
  val wordFeaturePair = reviewWordsPairs mapValues wordFeatures
  val avgWordFeaturesPair = wordFeaturePair mapValues avgWordFeatures
  val featuresPair = avgWordFeaturesPair join samplePairs mapValues {
    case (features, Sample(id, review, sentiment)) => LabeledPoint(sentiment.get.toDouble, features)
  }
  val trainingSet = featuresPair.values

  // Classificação
  println("\n<--- Aprendendo a Relação Entre as Palavras e Avaliando o Modelo --->\n")
  val Array(x_train, x_test) = trainingSet.randomSplit(Array(0.7, 0.3))
  val model = SVMWithSGD.train(x_train, 100)

  val result = model.predict(x_test.map(_.features))

  println(s"\n10 amostras:\n")
  x_test.map { case LabeledPoint(label, features) => s"$label -> ${model.predict(features)}" } take 10 foreach println
  val accuracy = x_test.filter(x => x.label == model.predict(x.features)).count.toFloat / x_test.count
  println(s"\n<--- Acurácia do Modelo: $accuracy --->")

  println("\n<---- Trabalho Concluído! ---->")
  Thread.sleep(10000)
}