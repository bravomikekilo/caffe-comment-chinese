#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}
/**Reshape(const vector<int>& shape); 按形状向量的整形函数 */
template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);            //检查是否超出最大维数
  count_ = 1;
  shape_.resize(shape.size());                     //调整形状向量的大小
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) { // 调整shape_data_ 的大小
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data()); //导出shape_data_ 的cpu端数据指针
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);               //保证每一个轴的范围非负
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";//保证总容量不超过int型的上限
    count_ *= shape[i];                  //累乘计算出总容量
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {              //更新容量(capacity_)的值
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));// 按容量分配data域空间
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));// 按容量分配diff域空间
  }
}
/**Reshape(const BlobShape& shape); 按BlobShape(由proto定义在caffe.pb.h中)的整形函数 */
template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) { // 将BlobShape转换为形状向量
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec); //以形状向量调用整形函数(Reshape)
}
/**ReshapeLike(const Blob<Dtype>& other) 按other的形状整形(Reshape)*/
template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}
// 构造函数组
/** 旧式构造函数*/
template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}
/** Blob(const vector<int>& shape) 按形状向量构造*/
template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}
//函数组结束
/** gpu_shape() 返回gpu端的形状向量 只读*/
template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

/** cpu_data() 返回cpu端的data域 只读*/
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

/** set_cpu_data() 设置cpu端data域数据*/
template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

/** gpu_data() 返回gpu端的data域 只读*/
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

/** cpu_diff() 返回cpu端的diff域 只读*/
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

/** gpu_diff() 返回gpu端的diff域 只读*/
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

/** mutable_cpu_data() 返回cpu端的data域 读写 */
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

/** mutable_gpu_data() 返回gpu端的data域 读写 */
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

/** mutable_cpu_diff() 返回cpu端的diff域 读写*/
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

/** mutable_gpu_diff() 返回gpu端的diff域 读写*/
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

/** ShareData() 共享Data域*/
template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

/** ShareDiff() 共享diff域*/
template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}
// Blob的update方法是使用在网络参数上的,所以只实现了Blob<float> Blob<double> 版本 
// Blob<int> Blob<unsigned int> 版本的没有实现
// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }
/** Update() 更新Blob中的数据 按照域的head属性确定数据在那一端超前,在超前端进行更新,当数据是同步的时更新在gpu端进行*/
template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized."; //未初始化的head属性将引发错误
  }
}
// Blob<unsigned int> 和 Blob<int> 的 data域L1正则函数没有实现
template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}
/**asum_data() 返回Blob的data域的L1正则 */
template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

// Blob<unsigned int> 和 Blob<int> 的 diff域L1正则函数没有实现
template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

/**asum_diff() 返回Blob的diff域的L1正则 计算将在超前的数据上进行 同步的情况下 计算在GPU端进行 未初始化的head属性将引发错误 */
template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

// Blob<unsigned int> 和 Blob<int> 的 data域L2正则函数没有实现
template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

/**sumsq_data() 返回Blob的data域的L2正则 计算将在超前的数据上进行 同步的情况下 计算在GPU端进行 未初始化的head属性将引发错误 */
template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

// Blob<unsigned int> 和 Blob<int> 的 data域L2正则函数没有实现
template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

/**sumsq_diff() 返回Blob的diff域的L2正则 计算将在超前的数据上进行 同步的情况下 计算在GPU端进行 未初始化的head属性将引发错误 */
template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

// Blob<unsigned int> 和 Blob<int> 的 data域比例缩放函数没有实现
template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

/**scale_data() Blob的data域的比例缩放函数 计算将在超前的数据上进行 同步的情况下 计算在GPU端进行 未初始化的head属性将引发错误 */
template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

// Blob<unsigned int> 和 Blob<int> 的 diff域比例缩放函数没有实现
template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

/**scale_diff() Blob的diff域的比例缩放函数 计算将在超前的数据上进行 同步的情况下 计算在GPU端进行 未初始化的head属性将引发错误 */
template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}
/**ShapeEquals(const BlobProto& other); 返回此Blob与other的形状是否相同 */
template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

/**CopyFrom(const Blob& source,bool copy_diff,bool reshape); 拷贝source的数据,copy_diff决定是否拷贝diff域，reshape决定是否整形 若source与此Blob不同型,且reshape=false,则会引发错误*/
template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}
/**FromProto(const BlobProto& proto,bool reshape); 反序列化函数读出proto中的数据保存在此Blob的CPU端上,在类型不匹配且reshape=false时会引发错误 */
template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

/**ToProto(BlobProto* proto,bool write_diff);序列化函数,将此blob中的数据写入proto中,write_diff决定是否写入diff域 写入的是CPU端的数据 注意数据通步 */
template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob); //宏操作 将Blob模板类在float和double上实例化
template class Blob<int>;         // 将Blob模板类在int 和 unsigned int 上实例化
template class Blob<unsigned int>;

}  // namespace caffe

