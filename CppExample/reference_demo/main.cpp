#include <iostream>
#include <string>
#include <map>
class Jun {
private:
    std::string name;
    int age;

public:
    std::string getName() {
        return name;
    }
    void setName(const std::string& value) {
        name = value;
    }
    int getAge() {
        return age;
    }
    void setAge(int& value) {
        age = value;
    }
    void display() {
        std::cout << "name: " << name << std::endl
                  << "age: " << age << std::endl;
    }
};
static std::map<std::string, Jun> name_jun_map;

Jun& get_ref_jun(const std::string& name) {
    bool flag = (name_jun_map.find(name) == name_jun_map.end());
    Jun& the_jun = name_jun_map[name];
    if (flag) {
        the_jun.setName(name);
    }
    return the_jun;
}

void test_ref_1(Jun& the_jun) {
  the_jun.setName("jun111");
}

//void test_ref_2(const Jun& the_jun) {
//  the_jun.setName("jun222");
//}

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::string name = "song";
    Jun& the_jun = get_ref_jun((std::string) "song");
    the_jun.display();
    return 0;
}
