#include <iostream>
#include <string>
#include <utility>

struct MyStruct {
  std::string name = "null";
  int age = 0;

  MyStruct() : name("empty"), age(1) {}
  MyStruct(std::string name, int age) :
    name(std::move(name)), age(age) {}
};

void printMyStruct(MyStruct mys) {
  std::cout << "name: " << mys.name
            << ", age: " << mys.age << std::endl;
}

int main() {
  std::cout << "Hello, World!" << std::endl;
  printMyStruct(MyStruct("songjun", 19));

  MyStruct mys;
  mys.name = "songjun";
  mys.age = 10;
  printMyStruct(mys);
}
