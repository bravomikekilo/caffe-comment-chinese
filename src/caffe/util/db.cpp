#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

#include <string>

namespace caffe { namespace db {
<<<<<<< HEAD
// 获得数据库的全局函数 调用对应后端的构造函数
DB* GetDB(DataParameter::DB backend) { //注意不是一个命名空间的DB
=======

DB* GetDB(DataParameter::DB backend) {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  switch (backend) {
#ifdef USE_LEVELDB
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (backend == "lmdb") {
    return new LMDB();
  }
#endif  // USE_LMDB
  LOG(FATAL) << "Unknown database backend";
  return NULL;
}

}  // namespace db
}  // namespace caffe
