/**
  * Author: songjun
  * Date: 2018/12/2
  */

import org.json.simple.JSONObject
import org.json.simple.parser.JSONParser
import org.json4s._
import org.json4s.native.JsonMethods

import scala.collection.JavaConversions._
object JsonTest {
  def main(args: Array[String]): Unit = {
    val jStr = "{\"name\":\"JunSong\", \"age\":20}"
    val jParser = new JSONParser()
    val jObj = jParser.parse(jStr).asInstanceOf[JSONObject]
    println(jObj.toString())
    println(jObj.size())
    for( k <- jObj.keySet()){
      println(s"$k : ${jObj.get(k)}")
    }


    val confMap = JsonMethods.parse(jStr).values.asInstanceOf[Map[String,String]]
    println(confMap.toString())

  }
}
