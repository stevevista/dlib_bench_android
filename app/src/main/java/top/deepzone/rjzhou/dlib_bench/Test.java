package top.deepzone.rjzhou.dlib_bench;

public class Test {
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    public static long convolutionTest() {
        return runTest("conv");
    }

    public static long alexnetTest() {
        return runTest("alexnet");
    }

    public static long resnetTest() {
        return runTest("resnet34");
    }
    public static long inceptionTest() {
        return runTest("inception");
    }
    private static native long runTest(String type);
}
