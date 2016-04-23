#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
//并行化头文件 将网络参数放置在安全的数据结构中
namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  /**构造函数 只能用求解器的共享指针构造*/
  explicit Params(shared_ptr<Solver<Dtype>> root_solver);
  virtual ~Params() { //虚析构函数
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers 缓冲区大小
  Dtype* data_;                 // Network parameters 网络参数
  Dtype* diff_;                 // Gradient 梯度

DISABLE_COPY_AND_ASSIGN(Params); //宏操作 取消Params类的赋值和拷贝操作符
};

// Params stored in GPU memory.
// GPUParams 储存在显存中的Params
template<typename Dtype>
class GPUParams : public Params<Dtype> { //共有继承
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);//用求解器的共享指针和  cuda设备号构造
  virtual ~GPUParams();
  /**configure(solver<Dtype>* solver) const; 配置函数*/
  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_; //属性 缓冲区大小
  using Params<Dtype>::data_; //属性 数据
  using Params<Dtype>::diff_; //属性 梯度
};

// DevicePair 设备对 将GPU按计算机的拓扑结构决定的亲密度按对组织
class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Synchronous data parallelism using map-reduce between local GPUs.
// P2PSync 端对端同步 在本地GPU之间用map-reduce实现同步的数据并行 
template<typename Dtype>
class P2PSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback, public InternalThread { //多重继承 
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype>> root_solver, P2PSync<Dtype>* parent, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  void Prepare(const vector<int>& gpus, vector<shared_ptr<P2PSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }

 protected:
  void on_start();
  void on_gradients_ready();

  void InternalThreadEntry();

  P2PSync<Dtype>* parent_;
  vector<P2PSync<Dtype>*> children_;
  BlockingQueue<P2PSync<Dtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
