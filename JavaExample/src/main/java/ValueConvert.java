import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Author: songjun
 * Date: 2018/10/24
 * Description:
 **/
public class ValueConvert {
    public static void main(String[] args){
        double d = 0.0000000123;
        String ds = String.valueOf(d);

        String ds1 = "1.23e-4";
        double d1 = Double.valueOf(ds1);

        System.out.printf("double: %f double string: %s%n",d, ds);
        System.out.printf("double string: %s, double: %f%n", ds1, d1);
    }
}
