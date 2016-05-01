#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

// Convert macro to string 将宏转换为字符串
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

/**DISABLE_COPY_AND_ASSIGN(classname) 宏操作 取消类的赋值和拷贝操作符*/
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

/**INSTANTIATE_CLASS(classname) 宏操作 将模板类在float和double类型下实例化*/
// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

/**INSTANTIATE_LAYER_GPU_FORWARD(classname) 宏操作 实例化float,double的Forward_gpu函数*/
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);
/**INSTANTIATE_LAYER_GPU_BACKWARD(classname) 宏操作 实例化float,double的Backward_gpu函数*/
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<Blob<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<Blob<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<double>*>& bottom)
/**INSTANTIATE_LAYER_GPU_FUNCS(classname) 宏操作 实例化float,double的Backward 和 Forward_gpu函数*/
#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

/**NOT_IMPLEMENTED 宏 中断程序并说明代码未实现*/
// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
/**Mat 引入opencv的Mat数据结构*/
namespace cv { class Mat; }

namespace caffe {
// 使用boost库的shared_ptr而不是新的C++11的特性 是因为CUDA和C++11的新特性不能良好兼容
// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;
//shared_ptr 共享指针 是共享所有权的智能指针 其特性是引用计数自动释放 注意：不能出现循环引用，否则会导致内存泄露
//导入常用的std标准库中的函数和类
// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

/**GlobalInit(int* pargc,char*** pargv); 全局初始化函数,在正式工作之前调用 按命令行参数进行全局初始化
 *目前是初始化google flags 和 google logging
 */
// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);
/**Caffe 类 一个单例类,包含常用的Caffe工具 像cublas curand 的句柄
  *单例类 目的是在一个线程中只有一个唯一的Caffe对象 协调cublas curands 等的局部上下文
  *
  */
// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();

  /**static Caffe& Get(); 单例类的关键 一个static函数 
    *在每一个线程中维持唯一的Caffe对象 保证线程的局部上下文 
    *具体实现在caffe/common.cpp 中
    */
  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static Caffe& Get();

  /** enum Brew {CPU , GPU}; 枚举 表示Caffe的工作模式*/
  enum Brew { CPU, GPU };

  /**RNG 随机数发生器类 将随机数发生器在boost和CUDA的实现的差异屏蔽掉 实现跨平台兼容性*/
  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
  /**cublas_handle() 获得cublas_handle_ */
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  /**curand_generator() 获得curand_generator_ */
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  /**mode() 返回工作模式 CPU或GPU*/
  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  //变量的设置函数
  // The setters for the variables
  /**set_mode() 设置工作模式 不推荐在程序中间更改工作模式 更改工作模式可能导致内存释放不正确*/
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }

  /**set_random_seed() 设置boost和curand的随机数种子*/
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);

  /**SetDevice(device_id) 唤醒(设置)设备 唤醒(设置)设备同时会重置cublas和curand*/
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);

  /**DeviceQuery() 设备查询打印出当前的gpu状态*/
  // Prints the current GPU status.
  static void DeviceQuery();

  /**CheckDevice(const int device_id); 返回device_id指明的设备是否可用*/
  // Check if specified device is available
  static bool CheckDevice(const int device_id);

  /**FindDevice(const int start_id = 0); 从start_id开始搜索设备 返回第一个可用设备的设备号*/
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);

  //打印并行训练信息
  // Parallel training info
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  /**root_solver() 返回是否是根求解器*/
  inline static bool root_solver() { return Get().root_solver_; }
  /**set_root_solver() 设置是否为根求解器*/
  inline static void set_root_solver(bool val) { Get().root_solver_ = val; }

 protected:
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_;       //属性 cublas句柄
  curandGenerator_t curand_generator_; //属性 curand_generator curand随机数发生器
#endif
  shared_ptr<RNG> random_generator_;   //属性 随机数发生器的指针

  Brew mode_;         //属性 工作模式
  int solver_count_;  //属性 求解器数量
  bool root_solver_;  //属性 是否是根求解器

 private:
  /**Caffe() 私有构造函数 防止破坏单例的条件*/
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
