//#include "test1.cpp"
#include <string>
#include <unordered_map>

class Test1 {
public:
  std::string name = "NULL";
  uint32_t age = 0;
};



int main() {
//  test1();
  std::unordered_map<std::string, Test1> name_to_t;
  Test1 t1;
  t1.name = "t1";
  t1.age = 10;
  name_to_t.insert(t1);

  Test1 t2;
  t2.name = "t2";
  t2.age = 2;

  name_to_t1.insert("songjun")


    return 0;
}
