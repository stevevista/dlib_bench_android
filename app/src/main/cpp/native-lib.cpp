#include <jni.h>
#include <string>
#include <dlib/dnn.h>
#include <iostream>

using namespace std;
using namespace dlib;

class loss_bench_
{
public:

    typedef unsigned long training_label_type;
    typedef unsigned long output_label_type;

    template <
            typename SUB_TYPE,
            typename label_iterator
    >
    void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
    ) const
    {
        // output nothing, just for benchmark
    }
};

template <typename SUBNET>
using bench_loss = add_loss_layer<loss_bench_, SUBNET>;
using bench_input = input_rgb_image_sized<227>;

static void randomize_input (
        matrix<rgb_pixel>& params
)
{
    dlib::rand rnd(std::rand());
    for (auto& val : params)
    {
        // Draw a random number to initialize the layer according to formula (16)
        // from Understanding the difficulty of training deep feedforward neural
        // networks by Xavier Glorot and Yoshua Bengio.
        val.red = rnd.get_random_float();
        val.green = rnd.get_random_float();
        val.blue = rnd.get_random_float();
    }
}

namespace CONV {

    using net_type = bench_loss<
            con<64,5,5,1,1,
                    bench_input
            >>;
}

namespace ALEX {

    using net_type = bench_loss<
            fc<10,
                    relu<fc<84,
                            relu<fc<120,
                                    max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                            max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                                    bench_input
                                            >>>>>>>>>>>>;
}


namespace RES34 {
// ----------------------------------------------------------------------------------------

// This block of statements defines the resnet-34 network

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using level1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
    template <typename SUBNET> using level2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
    template <typename SUBNET> using level3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
    template <typename SUBNET> using level4 = ares<64,ares<64,ares<64,SUBNET>>>;

    using net_type = bench_loss<fc<1000,avg_pool_everything<
            level1<
                    level2<
                            level3<
                                    level4<
                                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                                                    bench_input
                                            >>>>>>>>>>>;
}


namespace INCEPT {
// Inception layer has some different convolutions inside.  Here we define
// blocks as convolutions with different kernel size that we will use in
// inception layer block.
    template <typename SUBNET> using block_a1 = relu<con<10,1,1,1,1,SUBNET>>;
    template <typename SUBNET> using block_a2 = relu<con<10,3,3,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
    template <typename SUBNET> using block_a3 = relu<con<10,5,5,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
    template <typename SUBNET> using block_a4 = relu<con<10,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;

// Here is inception layer definition. It uses different blocks to process input
// and returns combined output.  Dlib includes a number of these inceptionN
// layer types which are themselves created using concat layers.
    template <typename SUBNET> using incept_a = inception4<block_a1,block_a2,block_a3,block_a4, SUBNET>;

// Network can have inception layers of different structure.  It will work
// properly so long as all the sub-blocks inside a particular inception block
// output tensors with the same number of rows and columns.
    template <typename SUBNET> using block_b1 = relu<con<4,1,1,1,1,SUBNET>>;
    template <typename SUBNET> using block_b2 = relu<con<4,3,3,1,1,SUBNET>>;
    template <typename SUBNET> using block_b3 = relu<con<4,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;
    template <typename SUBNET> using incept_b = inception3<block_b1,block_b2,block_b3,SUBNET>;

// Now we can define a simple network for classifying MNIST digits.  We will
// train and test this network in the code below.
    using net_type = bench_loss<
            fc<10,
                    relu<fc<32,
                            max_pool<2,2,2,2,incept_b<
                                    max_pool<2,2,2,2,incept_a<
                                            bench_input
                                    >>>>>>>>;
}

extern "C" JNIEXPORT jlong JNICALL
Java_top_deepzone_rjzhou_dlib_1bench_Test_runTest(
        JNIEnv *env,
        jobject /* this */,
        jstring type) {

    const char* str;
    str = env->GetStringUTFChars(type, nullptr);
    if(str == NULL) {
        return NULL; /* OutOfMemoryError already thrown */
    }
    std::string stype = str;

    matrix<rgb_pixel> img;
    img.set_size(227, 227);
    randomize_input(img);

    auto start_time = std::chrono::system_clock::now();

    if (stype == "conv")
    {
        CONV::net_type net;
        net(img);
    }
    else if (stype == "alexnet")
    {
        ALEX::net_type net;
        net(img);
    }
    else if (stype == "resnet34")
    {
        RES34::net_type net;
        net(img);
    }
    else if (stype == "inception")
    {
        INCEPT::net_type net;
        net(img);
    }
    auto eclipsed = std::chrono::system_clock::now() - start_time;
    jlong ret = static_cast<jlong>(std::chrono::duration_cast<std::chrono::milliseconds>(eclipsed).count());

    env->ReleaseStringUTFChars(type, str);
    return ret;
}

extern "C" JNIEXPORT jstring JNICALL
Java_top_deepzone_rjzhou_dlib_1bench_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
