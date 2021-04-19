import java.util.concurrent.*;

/**
 * @Author: liling
 * @Date: 2021/4/14 11:08 上午
 * @Description:
 */
public class MutiThead {
    private void runThread() throws Exception {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> stringFuture = executor.submit(new Callable<String>() {
            @Override
            public String call() throws Exception {
                Thread.sleep(2000);
                return "async thread";
            }
        });
        Thread.sleep(1000);
        System.out.println("main thread");
        System.out.println(stringFuture.get());
    }

    private void runCompletableFuture() throws Exception {
        // 创建异步执行任务，supplyAsync是有返回值的任务
        CompletableFuture<Double> cf = CompletableFuture.supplyAsync(MutiThead::fetchPrice);
        // 如果执行成功：
        cf.thenAccept((result) -> {
            System.out.println("price: " + result);
        });
        // 如果执行异常
        cf.exceptionally((e) -> {
            e.printStackTrace();
            return null;
        });
        // 主线程不能立即结束，否则CompletableFuture默认使用的线程池会立即关闭
        Thread.sleep(200);
    }

    static Double fetchPrice()
    {
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
        }
        if (Math.random() < 0.3) {
            throw new RuntimeException("fetch price failed");
        }
        return 5 + Math.random() * 20;
    }

    public static void main(String[] args) throws Exception {
        MutiThead mutiThead = new MutiThead();
        mutiThead.runCompletableFuture();
    }
}



