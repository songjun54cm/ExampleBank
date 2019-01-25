package SimpleExample;

/**
 * Author: songjun
 * Date: 2019/1/25
 * Description:
 **/
public class LongType {
    public static void main(String[] args){
        String valStr = "123456";
        Long val = Long.valueOf(valStr);
        Long v = (val % 100) / 10;
        System.out.println(v);
    }
}
