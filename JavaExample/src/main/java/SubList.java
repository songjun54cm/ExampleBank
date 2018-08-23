import java.util.ArrayList;
import java.util.List;
public class SubList {
    public static void main(String[] args){
        List<String> test = new ArrayList<String>();
        test.add("a");
        test.add("b");
        test.add("c");
        test.add("d");
        test.add("e");
        List<String> test1 = new ArrayList<String>();
        test1.add("ab");
        test1.add("abc");
        test.addAll(test1);
        printList("original list", test);
        List<String> sub = test.subList(1,3);
        printList("sub list", sub);
        List<String> sub1 = new ArrayList<String>(test.subList(1,test.size()));
        test.remove(1);
        printList("original list", test);
//        printList("sub1", sub);
        printList("sub2", sub1);

    }

    private static void printList(String listName, List<String> vals){
        System.out.print("the list: "+listName+ " ");
        for(String v : vals){
            System.out.print(v + " ");
        }
        System.out.println();
    }
}
