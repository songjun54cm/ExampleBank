#include <iostream>

bool IsOk(float* value) {
  *value = 0.5;
  return true;
}

int main() {
  float a = 0.0;
  bool ok = IsOk(&a);

  std::cout << "a: " << a << std::endl;

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
