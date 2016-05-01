#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
<<<<<<< HEAD
 * 将数据从源中读入数据层可用的队列中 一个源一个读取线程 即使是有多个求解器并行 保证数据库是串行读取的 
 */
 // 数据读取器类
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);// 构造函数
  ~DataReader();  //析构函数

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_; //队列释放
=======
 */
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
<<<<<<< HEAD
  // 队列对 队列对是在体和它的读取器之间共享的
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

<<<<<<< HEAD
    BlockingQueue<Datum*> free_; // 阻塞队列 多线程传递数据用
    BlockingQueue<Datum*> full_; // 阻塞队列 多线程传递数据用
=======
    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
