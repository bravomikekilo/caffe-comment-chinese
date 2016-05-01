#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"
//多线程的实现
namespace caffe {
/**~InternalThread() 析构函数 虚函数 默认调用StopInternalThread() 停止线程 */
InternalThread::~InternalThread() {
  StopInternalThread();
}

/**is_started() 返回线程是否已经开始*/
bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

/**must_stop() 急停函数 强力停止 应当测试在线程内部进行循环时能否退出*/
bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

/**StartInternalThread() 启动线程 当线程已经起动或已经终止时调用会引发错误 在调用前用is_started检查线程状态*/
void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));      //获得当前设备号
#endif
  Caffe::Brew mode = Caffe::mode();        //获得当前Caffe模式
  int rand_seed = caffe_rng_rand();        //获得随机数种子(rand_seed)
  int solver_count = Caffe::solver_count();//获得当前solver_count
<<<<<<< HEAD
  bool root_solver = Caffe::root_solver(); //获得当前root_solver
=======
  bool root_solver = Caffe::root_solver(); //获得当前root_solver 
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
//调用boost库启动线程
  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
<<<<<<< HEAD
          // 注意 this 指针在类方法调用时是C++ 自动加入 在这里要手动加入
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

/**entry(int device,Caffe::Brew mode,int rand_seed.int solver_count,bool root_solver); 私有的帮助函数
 *  用于启动线程后的本地状态初始化 用传入的参数初始化线程的本地状态
 */
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  //设置Caffe的全局状态
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);
  //运行子类中实现的程序
  InternalThreadEntry();
}
/** StopInternalThread() 终止内部线程 阻塞直到线程退出*/
void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
