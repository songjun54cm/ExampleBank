//
// Created by Jun Song on 2020/3/23.
// Authors: SongJun<songjun@kuaishou.com>
// Description: 
//

#include <iostream>
#include <map>
#include <vector>


void test_1_map(std::map<std::string, std::vector<std::string>>& the_map) {
  the_map["in_test_1"].push_back("test_1_0");
  the_map["in_test_1"].push_back("test_1_0");
}

void test_2_map(std::map<std::string, std::vector<std::string>>* the_map) {
  (*the_map)["in_test_2"].push_back("test_2_0");
  the_map->erase("in_test_1");
}

void display_map(const std::map<std::string, std::vector<std::string>>* the_map) {
  for (auto& kv : (*the_map)) {
    std::cout << "key:" << kv.first << std::endl
              << " values:";
    for (auto& i : kv.second) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
  }
}

void test1() {
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

  std::map<std::string, std::vector<std::string>> map1;
  map1["1"].push_back("111");
  map1["1"].push_back("1111");
  map1["2"].push_back("222");
  test_1_map(map1);
  display_map(&map1);
  test_2_map(&map1);
  display_map(&map1);
}