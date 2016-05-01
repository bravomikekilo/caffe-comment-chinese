/**一个层工厂 用来注册层
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
<<<<<<< HEAD
  typedef shared_ptr<Layer<Dtype>> (*Creator)(const LayerParameter&);// Creator 用层参数生成层的函数指针
  typedef std::map<string, Creator> CreatorRegistry;                 // 关系数组 Creator的登记字典

  static CreatorRegistry& Registry() { //Registry 注册函数 静态方法 生成一个Creator的登记字典
=======
  typedef shared_ptr<Layer<Dtype>> (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

<<<<<<< HEAD
  // Adds a creator. 向登记字典中加入一个 creator
=======
  // Adds a creator.
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

<<<<<<< HEAD
  // Get a layer using a LayerParameter. 由层参数获得一个层的指针
=======
  // Get a layer using a LayerParameter.
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
<<<<<<< HEAD
        << " (known types: " << LayerTypeListString() << ")"; // 未知的层类型 打印所有已知层类型
    return registry[type](param); // 调用层类型对应的Creator来生成层的指针
  }
  
  // 静态方法 输出包含(登记字典中所有的层的类型的)向量
=======
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
<<<<<<< HEAD
  // static variables. 私有的构造函数 单例类的构造模式
  LayerRegistry() {}
  // 将包含层类型的向量转化成字符串
  static string LayerTypeListString() { 
=======
  // static variables.
  LayerRegistry() {}

  static string LayerTypeListString() {
>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};

<<<<<<< HEAD
// 层注册器类 构造时将层注册
=======

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

<<<<<<< HEAD
// 宏定义 将层注册 注意 type 不要加双引号
=======

>>>>>>> 69d9c2663b93a3129d1c8d044ef04546546955b6
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
