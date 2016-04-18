#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**StartInternalThread() 启动线程 当线程已经启动或者已经终止时调用会引发错误 在调用前用is_started检查线程状态
   * Caffe的线程的本地状态将用当前的线程值来初始化
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();
  /** StopInternalThread() 终止内部线程 阻塞直到线程退出*/
  /** Will not return until the internal thread has exited. */
  void StopInternalThread();
  /** is_started() 返回线程是否已经开始 */
  bool is_started() const;

 protected:
  /**InternalThreadEntry() 内部线程入口函数 虚函数 子类通过实现此函数来实现运行不同代码 */
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  /**must_stop() 急停函数 强力停止 应当测试在线程内部进行循环时能否退出*/
  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;//属性 指向boost线程对象的共享指针
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
