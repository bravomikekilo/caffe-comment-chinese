#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
<<<<<<< HEAD
  string key = source_key(param); // 获得资源的key 
  weak_ptr<Body>& weak = bodies_[key]; // 按关键字查登记字典
  body_ = weak.lock(); // 获得body的共享指针
  if (!body_) { // 如果body不存在 即资源第一次被访问 创建body
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_); // 将自己压入body 的队列中
=======
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
<<<<<<< HEAD
  body_.reset(); //将共享指针重置 如果这是最后一个引用 body将被释放
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) { // 如果对应的body已经释放 从登记字典中删除记录
=======
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
<<<<<<< HEAD
  // 构造函数 将free_初始化 填入相应数目的数据
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
<<<<<<< HEAD
  // 将两个队列循环弹栈 全部清空 
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
<<<<<<< HEAD
    // 为保证的运行的确定性 只在所有求解器均准备好的时候启动运行
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
<<<<<<< HEAD
  // 希望能就地反序列化
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
<<<<<<< HEAD
  if (!cursor->valid()) { // 不合法的坐标
=======
  if (!cursor->valid()) {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

}  // namespace caffe
