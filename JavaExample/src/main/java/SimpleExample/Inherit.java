package SimpleExample;

import java.util.*;
public class Inherit {
    public static void main(String[] args){
        SubClass1 s1 = new SubClass1();
        SubClass2 s2 = new SubClass2();
        System.out.println("SimpleExample.SubClass1 name: " + s1.getName());
        System.out.println("SimpleExample.SubClass1 sex: " + s1.getSex());
        System.out.println("SimpleExample.SubClass1 static: " + s1.getStaticStr());
        System.out.println("SimpleExample.SubClass2 name: " + s1.getName());
        System.out.println("SimpleExample.SubClass2 sex: " + s1.getSex());
        System.out.println("SimpleExample.SubClass2 static: " + s1.getStaticStr());

        Set<String> temp = new HashSet<String>();
        s1.fillSet(temp);
        for(String s : temp){
            System.out.println(s);
        }
    }
}

abstract class SuperClass {
    static String staticStr = "Super Static";
    public String label="Super";
    public  String Name="Super";
    public final String Sex="Super sex";
    protected String abc;
    SuperClass(){
        System.out.println("SimpleExample.SuperClass()");
    }
    SuperClass(String label) {
        System.out.println("SimpleExample.SuperClass(label)");
        this.label = label;
    }
    public String getName(){
        return Name;
    }
    public String getSex(){
        return Sex;
    }
    public String getStaticStr(){return staticStr;}

}

class SubClass1 extends SuperClass{
    static String staticStr = "Sub1 Static";
    private String label="Sub1 label";
    private  String Name="Sub1 Name";
    private final String Sex="Sub1 sex";
    SubClass1(){
        System.out.println("SimpleExample.SubClass1");
    }

    public SubClass1(String label){
        super("inSub1");
        System.out.println("SimpleExample.SubClass1(label):"+label);
        this.label = label;
    }

    public void fillSet(Set<String> aSet){
        aSet.add("a");
        aSet.add("b");
    }
}

class SubClass2 extends SuperClass{
    static String staticStr = "Sub2 Static";
    private String label="Sub2 label";
    private  String Name="Sub2 Name";
    private final String Sex="Sub2 sex";
    SubClass2(){
        System.out.println("SimpleExample.SubClass2");
    }

    public SubClass2(String label){
        super("inSub2");
        System.out.println("SimpleExample.SubClass2(label):"+label);
        this.label = label;
    }
}