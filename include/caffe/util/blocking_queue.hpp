#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>

<<<<<<< HEAD
// 阻塞队列
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
namespace caffe {

template<typename T>
class BlockingQueue {
 public:
<<<<<<< HEAD
  explicit BlockingQueue(); //默认构造函数

  void push(const T& t); //压入
=======
  explicit BlockingQueue();

  void push(const T& t);
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
<<<<<<< HEAD
  // 窥视 返回元素但不移除
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  T peek();

  size_t size() const;

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
<<<<<<< HEAD
  class sync; //同步类

  std::queue<T> queue_;    //标准队列
  shared_ptr<sync> sync_;  //同步
=======
  class sync;

  std::queue<T> queue_;
  shared_ptr<sync> sync_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe

#endif
