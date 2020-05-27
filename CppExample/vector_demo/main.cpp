#include <iostream>
#include <map>
#include <vector>
int main() {
  std::cout << "Hello, World!" << std::endl;
  std::vector<int> vec{1,2,3,4,5,6,7,8,9,10};
  std::cout << vec.size() << "," << *vec.begin() << "," << *(vec.end()-1)
            << *(vec.begin() + 1) << "," << *(vec.begin() + vec.size() - 1) << std::endl;

  std::vector<int> new_vec(vec.begin(), vec.begin()+vec.size());
  std::cout << new_vec.size() << "," << *new_vec.begin() << "," << *(new_vec.end()-1) << std::endl;

  std::vector<std::map<int, int>> map_vec(2, std::map<int,int>());

  map_vec[0][1]=1;
  map_vec[0][12]=12;

  map_vec[1][2] = 2;
  map_vec[1][22] = 22;
  map_vec[1][222] = 222;

  std::cout << map_vec[0].size() << std::endl;
  std::cout << map_vec[1].size() << std::endl;

  return 0;


}
