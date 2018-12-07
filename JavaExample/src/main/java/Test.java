import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
public class Test {
    public static void main(String[] args){
        String exampleString = "example";
        InputStream stream = new ByteArrayInputStream(exampleString.getBytes(StandardCharsets.UTF_8));
    }

    public static void main1(String[] args){
        List<String> ls = new ArrayList<String>();
        List<String> ls1;
        ls.add("a");
        ls.add("b");
        ls.add("c");

        ls1 = ls.subList(0,Math.min(10, ls.size()));
        System.out.println("ls1 size: " + String.valueOf(ls1.size()));

        Map<String, List<String>> testMap = new HashMap<String, List<String>>();
        testMap.put("abc", new ArrayList<String>());
        testMap.get("abc").add("hello");
        for(String key: testMap.keySet()){
            System.out.println("key: " + key);
            for(String s: testMap.get(key)){
                System.out.println("val: " + s);
            }

        }
    }
}
