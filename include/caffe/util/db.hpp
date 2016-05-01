#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
<<<<<<< HEAD
// 数据库头文件 
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6

namespace caffe { namespace db {

enum Mode { READ, WRITE, NEW };

<<<<<<< HEAD
// 光标类
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

<<<<<<< HEAD
// 事物
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

<<<<<<< HEAD
// 数据库类 抽象基类
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_AND_ASSIGN(DB);
};

<<<<<<< HEAD
// 获得数据库的全局函数
=======
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
