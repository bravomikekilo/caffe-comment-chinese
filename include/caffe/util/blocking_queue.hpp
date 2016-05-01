#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>

// 阻塞队列
namespace caffe {

template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(); //默认构造函数

  void push(const T& t); //压入


  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
  // 窥视 返回元素但不移除
  T peek();

  size_t size() const;

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
  class sync; //同步类

  std::queue<T> queue_;    //标准队列
  shared_ptr<sync> sync_;  //同步

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe

#endif
