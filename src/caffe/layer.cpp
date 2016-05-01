#include <boost/thread.hpp>
#include "caffe/layer.hpp"
//基类Layer的实现
namespace caffe {

/**InitMutex() 初始化前向互斥体 */
template <typename Dtype>
void Layer<Dtype>::InitMutex() { 
  forward_mutex_.reset(new boost::mutex());
}

/**Lock() 前向互斥体上锁 */
template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

/**Unlock() 前向互斥体解锁释放 */
template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

INSTANTIATE_CLASS(Layer); //宏操作 将模板类Layer在float和double上实例化 也就是说 只有float和double的Layer

}  // namespace caffe
