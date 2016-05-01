#ifndef CPU_ONLY
#include <cuda_runtime.h> //引用cuda_runtime头文件
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream> //引用标准库字符串流 常用于格式转换
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

/**Op 枚举 操作类型*/
enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

/**apply_buffers 函数模板 应用缓冲区 按Op的指示应用缓冲区*/
template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs, Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
<<<<<<< HEAD
        // 用当前blob的值初始化缓冲区
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
        caffe_copy(size, reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()), ptr);//高危的类型转换
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);//将blob的data域的CPU端数据指针指向缓冲区
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);//将blob的data域的GPU端数据指针指向缓冲区
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);//将blob的diff域的CPU端的数据指针指向缓冲区
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);//将bolb的diff域的GPU端的数据指针指向缓冲区
        break;
    }
    ptr += size;//将指针移到数据末端
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));//检查总大小 total_size至少为一个字节
}

// Buffer size necessary to store given blobs
// 储存blobs所需要的必要缓冲区大小
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
<<<<<<< HEAD
  // Size have at least one byte, otherwise cudaMalloc fails if net has no 大小至少为一字节
=======
  // Size have at least one byte, otherwise cudaMalloc fails if net has no 大小至少为一字节 
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  //learnable parameters. 否则cudaMalloc会在网络无可学习参数的情况下出错
  return (size > 0) ? size : 1;
}

//Params类 的构造函数
template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
}

//GPUParams类 的构造函数
template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
<<<<<<< HEAD
      root_solver->net()->learnable_params();//从根求解器拷贝blob的值
=======
      root_solver->net()->learnable_params();
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

//GPUParams的析构函数
template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

//GPUParams的配置函数
template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

<<<<<<< HEAD
/**compute 将GPU按对排好 组织成树的形式*/
=======
/**compute */
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        cudaDeviceProp a, b;
        CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
        CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
        if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        CUDA_CHECK(
            cudaDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
<<<<<<< HEAD
  // 经过计算配对后 将只有根节点剩余
  CHECK_EQ(remaining.size(), 1);

  //将根节点插入向量头
=======
  CHECK_EQ(remaining.size(), 1);

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#else
  NO_GPU;
#endif
}

//
/**P2PSync类的构造函数*/
template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
<<<<<<< HEAD
    : GPUParams<Dtype>(root_solver, param.device_id()),//在参数指定的GPU上分配缓冲区
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),//获得起始的迭代次数
=======
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
<<<<<<< HEAD
  const int self = param.device_id();//获得设备号
  CUDA_CHECK(cudaSetDevice(self));   //启用设备

  if (parent == NULL) {//当无祖先时
    solver_ = root_solver; //将求解器设为根求解器
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));//将求解器设为Workersolver
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get()); //配置缓冲区
  solver_->add_callback(this);    //向求解器加入回调函数

  if (parent) {//当有祖先时
    // Enable p2p access between devices
    // 启用设备间的端对端通讯 (到祖先的通讯)
=======
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
<<<<<<< HEAD
    // 在祖先设备上分配接受缓冲
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

/**P2PSync 的析构函数*/
template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

/**InternalThreadEntry() 实现多线程的操作*/
template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
<<<<<<< HEAD
  Caffe::set_root_solver(false); //设置为子求解器
=======
  Caffe::set_root_solver(false);
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
<<<<<<< HEAD
  // 如果有祖先 等待祖先的更新
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();//祖先出栈
    CHECK(parent == parent_);             //检查祖先是否出现了混乱
  }

  // Update children
  // 更新孩子 设置数据传输的指针
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_; //src源指针
    Dtype* dst = children_[i]->data_; //dst 汇指针
=======
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

<<<<<<< HEAD
// 拷贝并同步内存
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));//异步数据拷贝
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));//流同步
    children_[i]->queue_.push(this);      //将自己压入孩子的栈
=======
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
<<<<<<< HEAD
  // 将出现在队列里的孩子的梯度求和
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif
<<<<<<< HEAD
//  gpu端求和
=======

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
<<<<<<< HEAD
  // 将梯度发往祖先
  if (parent_) { //如果有祖先
=======
  if (parent_) {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif
<<<<<<< HEAD
// 数据发送
=======

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
<<<<<<< HEAD
} else { //如果是根求解器分解梯度
=======
  } else {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}
/**Prepare() P2PSync 准备*/
template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
<<<<<<< HEAD
  // 为了map-reduce同步将设备结对
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  //将除了根设备以外的设备按对显示输出
=======
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

<<<<<<< HEAD
  SolverParameter param(solver_->param());//取出求解器的参数
=======
  SolverParameter param(solver_->param());
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

  //通过找到solver的祖先构建GPU树
  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}
/**Run() 按GPU向量并行运行*/
template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

INSTANTIATE_CLASS(Params);   //宏操作 将模板类Params 在float double下实例化
INSTANTIATE_CLASS(GPUParams);//宏操作 将模板类GPUParams 在float double下实例化
INSTANTIATE_CLASS(P2PSync);  //宏操作 将模板类P2PSync 在float double下实例化

}  // namespace caffe
