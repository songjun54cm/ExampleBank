#include <iostream>
#include <map>
#include <vector>


int main() {
    std::cout << "Hello, Map Demo" << std::endl;
    std::map<int, std::vector<int>> int_intVec;
    int_intVec[1].push_back(1);
    int_intVec[1].push_back(11);
    int_intVec[2].push_back(2);
    int_intVec[3].push_back(3);

    for (auto& kv : int_intVec) {
        std::cout << "key:" << kv.first << " values:";
        for (auto& i : kv.second) {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }


    return 0;
}
