#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

/**blob SyncedMemory 的封装 作为'层'(Layer)'网络'(Net)'求解器'(Solver) 之间的计算基本单位
 * 此类取消了拷贝和赋值操作符
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {} //blobs分为四个域 data diff count capacity
//构造函数组
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape); //按形状向量构造
//函数组结束
  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**Reshape 更改blob的维(轴)数和范围  如有必要将分配新的内存
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape); // 以形状向量调用
  void Reshape(const BlobShape& shape); // 以BlobShape(由proto定义在caffe.pb.h中)调用
  void ReshapeLike(const Blob);
  /**shape_string 将形状属性转换为字符串 形如"a b c d (count_)" */
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  /**shape() 返回形状属性(以形状向量的形式) */
  inline const vector<int>& shape() const { return shape_; }

  /**shape(int index) 返回索引指定轴的维数 如索引为负则倒着数 形如python
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  /**num_axes() 返回轴数 */
  inline int num_axes() const { return shape_.size(); }
  /**count() 返回数量 */
  inline int count() const { return count_; }

  /**count(int start_axis,int end_axis) const; 返回一个分片的容量 是一组轴的总容量
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**count(int start_axis) const; 返回一个分片的容量,由开始轴到最后的轴的总容量 start_axis 开始轴
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**CanonicalAxisIndex(int axis_index); 返回规范化的轴序号 -1是最后一个 同python
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() + index),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }
//旧函数组
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
//旧函数组
  /**offset(..) 计算偏移地址(寻址) */
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**CopyFrom(const Blob<Dtype>& source, bool copy_diff ,bool reshape); 拷贝函数 copy_diff reshape 决定是否拷贝diff域和进行reshape
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
  //
  /**data_at data域下标取值函数 */
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  /**data_at data域下标取值函数 */
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  /**diff_at diff域下标取值函数 */
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  /**diff_at diff域下标取值函数 */
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  /**data() 返回data域的syncedMemory */
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  /**diff() 返回diff域的syncedMemory */
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

 //数据存取函数组
  /** cpu_data() 返回cpu端的data域 只读*/
  const Dtype* cpu_data() const;

  /** set_cpu_data() 设置cpu端data域数据*/
  void set_cpu_data(Dtype* data);

  /** gpu_shape() 返回gpu端的形状向量 只读*/
  const int* gpu_shape() const;

  /** gpu_data() 返回gpu端的data域 只读*/
  const Dtype* gpu_data() const;

  /** cpu_diff() 返回cpu端的diff域 只读*/
  const Dtype* cpu_diff() const;

  /** gpu_diff() 返回gpu端的diff域 只读*/
  const Dtype* gpu_diff() const;

  /** mutable_cpu_data() 返回cpu端的data域 读写 */
  Dtype* mutable_cpu_data();

  /** mutable_gpu_data() 返回gpu端的data域 读写 */
  Dtype* mutable_gpu_data();

  /** mutable_cpu_diff() 返回cpu端的diff域 读写*/
  Dtype* mutable_cpu_diff();

  /** mutable_gpu_diff() 返回gpu端的diff域 读写*/
  Dtype* mutable_gpu_diff();
//函数组结束

  void Update();
  /** FromProto(const BlobProto& proto,bool reshape =true) 反序列化函数 从Proto文件中读出Blob的数据 reshape 决定是否整形*/
  void FromProto(const BlobProto& proto, bool reshape = true);
  /** ToProto(BlobProto *proto,bool write_diff = false) const; 序列化函数 将Blob数据写入Proto文件中 write_diff 决定是否写入diff域*/
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  /** asum_data() 计算data域中值的绝对值的和 */
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;

  /** asum_diff() 计算diff域中值的绝对值的和 */
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /** sumsq_data() 计算data域中值的平方和 */
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /** sumsq_diff() 计算diff域中值的平方和 */
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;
  /** scale_data(Dtype scale_factor) 按因子scale_factor 对data域内每个数据按比例缩放*/
  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor); 
  /** scale_diff(Dtype scale_factor) 按因子scale_factor 对diff域内每个数据按比例缩放*/
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor); 

  /**ShareData(const Blob& other) 将other的data域指针指向此Blob的data域
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);

  /**ShareDiff(const Blob& other) 将other的diff域的指针指向此Blob的diff域
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  /**ShapeEquals 判断两Blob是否同形状 */
  bool ShapeEquals(const BlobProto& other);

 protected:
  shared_ptr<SyncedMemory> data_; //属性 保存data域的 SyncedMemory
  shared_ptr<SyncedMemory> diff_; //属性 保存diff域的 SyncedMemory
  shared_ptr<SyncedMemory> shape_data_; //属性 保存形状数据(shape_data_)的 SyncedMemory
  vector<int> shape_; //属性 形状向量
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob); //宏操作 取消Blob类的拷贝和赋值操作符
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
