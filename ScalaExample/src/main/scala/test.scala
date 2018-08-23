import scala.util.matching.Regex
import java.text.SimpleDateFormat
import java.util.Calendar

object Test{
	def main(args: Array[String]) {
		val pattern = "\\$today([\\+\\-]\\d+)*".r
		val a = "123songjun$today-1/songjun$today+13songjun$today/dddd"
		println(pattern.findAllIn(a).mkString(","))
		// val today = "2018-01-01"

		var newS = a
		var si = 0
		var ei = a.length

		for (s <- pattern.findAllIn(a)){
			println(s)
			val ii = newS.indexOf(s)
			println(ii)
			newS = newS.substring(0, ii) + parseDate(s) + newS.substring(ii+s.length, newS.length)
			println(newS)
		}
		if (si < ei){
			newS = newS + a.substring(si, ei)
		}
		println(newS)
	}

	def parseDate(dateStr: String): String = {
		val dateFormat = new SimpleDateFormat("yyyy-MM-dd")
		val localTs = Calendar.getInstance().getTime().getTime()
		if(dateStr == "$today"){
			val todayStr = dateFormat.format(localTs)
			return todayStr
		}else if (dateStr.startsWith("$today+")){
			val delta = dateStr.substring(7).toInt
			val ts = localTs + 3600*24*delta* 1000L
			val newDateStr = dateFormat.format(ts)
			return newDateStr
		}else if (dateStr.startsWith("$today-")){
			val delta = dateStr.substring(7).toInt
			val ts = localTs - 3600*24*delta* 1000L
			val newDateStr = dateFormat.format(ts)
			return newDateStr
		}else{
			return "error"
		}
	}
}