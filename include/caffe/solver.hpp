#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {

/** Enum 枚举 用以表明Solver的用户的请求 这一机制允许在程序收到中断信号后保存快照
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**SolverAction 回调函数的类型
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**Solver 类 是优化网络的类的接口
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  //构造函数组
  explicit Solver(const SolverParameter& param, const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  //函数组结束
  /**Init(const SolverParameter& param); 初始化函数 初始化求解器 SolverParameter 在caffe/proto/caffe.bp.h中定义*/
  void Init(const SolverParameter& param);
  /**InitTrainNet(); Init的帮助函数*/
  void InitTrainNet();
  /**InitTestNets() Init的帮助函数*/
  void InitTestNets();

  /**SetActionFunction 设置回调函数*/
  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();

  /**Solve()  求解器函数的主入口*/
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  // 求解器函数的主入口 默认，迭代次数是0 为一个预训练过的网络传入一个非零的迭代次数
  // 只有根求解器才能调用Solve()函数
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);

  /**Restore() 重置函数*/
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);

  /**Snapshot() 快照函数 快照网络的学习状态*/
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  virtual ~Solver() {}
  /**param() 返回求解器参数*/
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  /**iter() 返回迭代状态*/
  int iter() { return iter_; }

  //Callback 回调类 在迭代的一个特定的点调用
  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  //检查快照是否可以写入
  void CheckSnapshotWritePermissions();
  /**type() 返回求解器的类型
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

 protected:
  /**为当前迭代计算并应用更新值 */
  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  /**TestAll() 测试流程*/
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  /**SnapshotSolverState() 快照求解器状态*/
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_; //属性 求解器参数
  int iter_;              //属性 迭代状态(次数)
  int current_step_;      //属性 当前步骤
  shared_ptr<Net<Dtype> > net_; //属性 网络的指针
  vector<shared_ptr<Net<Dtype> > > test_nets_; //属性 测试网络的指针
  vector<Callback*> callbacks_; //属性 回调函数的向量
  vector<Dtype> losses_;        //属性 误差向量
  Dtype smoothed_loss_;         //属性 平滑的误差


  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;//属性 指向根求解器的指针

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;//属性 回调函数

  // True iff a request to stop early was received.
  bool requested_early_exit_;//属性 是否被要求提前返回

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**WorkerSolver 类 Solver的子类 仅计算梯度的求解器 作为多GPU情况下的worker 不支持快照功能
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
