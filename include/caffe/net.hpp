#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**网络类 将层(Layer)连接在一起组成一个由NetParameter说明的有向无环图
 * NetParameter 在caffe/proto/caffe.pb.h 中定义的类
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, const Net* root_net = NULL);
  explicit Net(const string& param_file, Phase phase,
      const Net* root_net = NULL);
  virtual ~Net() {}

  /**Init(const NetParameter& param) 初始化用一个网络参数初始化网络 */
  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**Forward(Dtype* loss = NULL); 返回前向运行网络的结果
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**Forward Backward From To 是按照网络的拓扑顺序进行的
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**ClearParamDiffs 将所有网络参数的diff域清零 应当在调用backward之前调用
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  /**Backward()  */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /**name() 返回网络的名称 */
  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /**layer_names() 返回包含所有层的名称的向量 */
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /**blob_name() 返回包含所有Blob的名称的向量 */
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /**blobs() 返回所有的blob */
  /// @brief returns the blobs
<<<<<<< HEAD
  inline const vector<shared_ptr<Blob<Dtype>>>& blobs() const {
=======
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    return blobs_;
  }
  /**layers() 返回所有的层 */
  /// @brief returns the layers
<<<<<<< HEAD
  inline const vector<shared_ptr<Layer<Dtype>>>& layers() const {
    return layers_;
  }
  /**phase() 返回相 训练相或者测试相 */
=======
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /**phase() 返回相 训练相或者测试相 */ 
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }

  /**bottom_vecs() 返回每一层的bottom向量 */
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
<<<<<<< HEAD
  inline const vector<vector<Blob<Dtype>*>>& bottom_vecs() const {
=======
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    return bottom_vecs_;
  }
  /**top_vecs() 返回每一层的top向量
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
<<<<<<< HEAD
  /**params_lr() 返回可学习参数的学习因子 */
=======
  /**params_lr() 返回可学习参数的学习因子 */ 
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /**params_weight_decay() 返回可学习参数的衰减因子 */
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  /**param_names_index() 返回参数名的索引(关系数组) */
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  /**param_owners() 返回参数的所有者 */
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /**num_inputs() 返回输入blob的数目 */
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  /**num_outputs() 返回输出blob的数目 */
  inline int num_outputs() const { return net_output_blobs_.size(); }
  /**input_blobs() 返回网络的输入blob向量 */
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  /**output_blobs() 返回网络的输出blob向量 */
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  /**input_blob_indices()  */
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  /**output_blob_indices() */
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  /**has_blob(const string& blob_name) Blob查询 查询是否有名字为blob_name的blob  */
  bool has_blob(const string& blob_name) const;
  /**blob_by_name(const string& blob_name) Blob查询 返回名字为blob_name的blob */
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
<<<<<<< HEAD

=======
  
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  /**has_layer(const string& layer_name); Layer查询 查询是否有名字为layer_name的Layer  */
  bool has_layer(const string& layer_name) const;
  /**layer_by_name(const string& layer_name); Layer查询 返回名字为layer_name的Layer */
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;
  /**set_debug_info(const bool value); 设置debug信息*/
  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.Init初始化函数的帮助函数
<<<<<<< HEAD
  /**FilterNet(const NetParameter& param,NetParameter* param_filtered);
=======
  /**FilterNet(const NetParameter& param,NetParameter* param_filtered); 
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
   * 将param中描述的层从param_filtered 中滤除
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /**StateMeetsRule() 返回NetState是否符合NetStateRule的规则*/
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

 protected:
  // Helpers for Init.Init 函数的帮助函数
  /**AppendTop 在网络上追加一个输出(top) blob */
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /**AppendBottom 在网络上追加一个输入(bottom) blob */
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /**AppendParam 在网络上追加一个参数(parameter) blob */
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /**ForwardDebugInfo 在前向过程中显示调试信息的帮助函数 */
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /**BackwardDebugInfo 在后向过程中显示调试信息的帮助函数 */
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);

  /**UpdateDebugInfo 在更新过程中显示调试信息的帮助函数 */
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_; //属性 网络名称
  /// @brief The phase: TRAIN or TEST
  Phase phase_; //属性 相 训练相或者测试相
  /// @brief Individual layers in the net
<<<<<<< HEAD
  vector<shared_ptr<Layer<Dtype>>> layers_;//属性 层 储存所有层的共享指针的向量
=======
  vector<shared_ptr<Layer<Dtype> > > layers_;//属性 层 储存所有层的共享指针的向量
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  vector<string> layer_names_;          //属性 层的名称 储存属于此网络的所有层的名称的向量
  map<string, int> layer_names_index_;  //属性 层名称的索引 储存层的名称和向量下标的关系数组
  vector<bool> layer_need_backward_;    //属性 需要后向的层 存储是否需要后向的布尔向量
  /// @brief the blobs storing intermediate results between the layer.
<<<<<<< HEAD
  vector<shared_ptr<Blob<Dtype>>> blobs_;//属性 blobs_ 储存中间结果的blob的向量
=======
  vector<shared_ptr<Blob<Dtype> > > blobs_;//属性 blobs_ 储存中间结果的blob的向量
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  vector<string> blob_names_;        //属性 blob的名称 储存所有blob的名称的向量
  map<string, int> blob_names_index_;//属性 blob的名称的索引 储存blob的名称和向量下标的关系数组
  vector<bool> blob_need_backward_;  //属性 需要后向的bolb 储存是否需要后向的布尔向量
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*>> bottom_vecs_;
  vector<vector<int>> bottom_id_vecs_;
  vector<vector<bool>> bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
<<<<<<< HEAD
  vector<vector<Blob<Dtype>*>> top_vecs_;
  vector<vector<int>> top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int>> param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int>> param_layer_indices_;
=======
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
<<<<<<< HEAD
  vector<shared_ptr<Blob<Dtype>>> params_;
=======
  vector<shared_ptr<Blob<Dtype> > > params_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_; //可学习参数的id
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_; //属性 可学习参数的学习因子
  vector<bool> has_params_lr_; //属性 是否拥有学习因子
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_; //属性 权重的衰减因子
  vector<bool> has_params_decay_; //属性 是否拥有衰减因子
  /// The bytes of memory used by this net
  size_t memory_used_; //属性 网络对象占用的内存大小
  /// Whether to compute and display debug info for the net.
  bool debug_info_; //属性 是否计算并显示网络的调试信息
  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_; //属性 根网络 在数据并行中实际拥有共享层的网络
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
