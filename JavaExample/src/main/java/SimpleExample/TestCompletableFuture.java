package SimpleExample;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.*;
import java.util.function.Supplier;

/**
 * Author: songjun
 * Date: 2018/12/24
 * Description:
 **/
public class TestCompletableFuture {
    private final Executor testExecutor = new ThreadPoolExecutor(3, 6, 120L,
            TimeUnit.SECONDS, new LinkedBlockingQueue<>());

    public Long getSum(Long v){
        Long s = 0L;
        for(Long i=1L; i<=v; ++i){
            s = s + i;
        }
        return s;
    }

    public Set<Long> getSums1(Set<Long> valSet){
        Set<Long> resSet = new HashSet<>();
        for(Long v : valSet){
            resSet.add(getSum(v));
        }
        return resSet;
    }

    public Set<Long> getSums2(Set<Long> valSet){
        Set<Long> res = new HashSet<>();
        Set<CompletableFuture<Long>> futureSet = new HashSet<>();
        try{
            for(Long n : valSet){
                CompletableFuture<Long> future = CompletableFuture.supplyAsync(
                        new Supplier<Long>() {
                            @Override
                            public Long get() {
                                return getSum(n);
                            }
                        }, testExecutor);
                futureSet.add(future);
            }
            CompletableFuture<Void> allFuture = CompletableFuture.allOf(futureSet.toArray(new CompletableFuture[futureSet.size()]));
            allFuture.get(20, TimeUnit.SECONDS);
            for(CompletableFuture<Long> f : futureSet){
                res.add(f.getNow(null));
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return res;
    }

    public static void main(String[] args){
        Set<Long> valSet = new HashSet<>();
        for(int i=0; i < 300; ++i){
            valSet.add(1000000L+i);
        }
        TestCompletableFuture ins = new TestCompletableFuture();
        System.out.printf("valSet size: %d%n", valSet.size());
        // getSums1 time
        long startTime = System.currentTimeMillis();
        Set<Long> res1 = ins.getSums1(valSet);
        long endTime = System.currentTimeMillis();
        long useTime = endTime - startTime;
        System.out.printf("sum1: use time : %d, get %d res.%n", useTime, res1.size());

        // getSum2 time
        long startTime1 = System.currentTimeMillis();
        Set<Long> res2 = ins.getSums2(valSet);
        long endTime1 = System.currentTimeMillis();
        long useTime1 = endTime1 - startTime1;
        System.out.printf("sum2: use time : %d, get %d res.%n", useTime1, res2.size());
    }
}
