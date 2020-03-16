#include <iostream>
#include <time.h>
#include <string>

int main() {
  std::cout << "Hello, World!" << std::endl;

  time_t m_time = time(NULL);
  tm* ltm = localtime(&m_time);
  std::cout << "year: " << ltm->tm_year + 1900 << std::endl
            << "month: " << ltm->tm_mon << std::endl
            << "week: " << ltm->tm_wday << std::endl
            << "day: " << ltm->tm_mday << std::endl
            << "year day: " << ltm->tm_yday << std::endl
            << "hour: " << ltm->tm_hour << std::endl
            << "minute: " << ltm->tm_min << std::endl
            << "second: " << ltm->tm_sec << std::endl
            << "is dst: " << ltm->tm_isdst << std::endl;

  char cstr[80];
  strftime(cstr, 100, "%Y年%m月%d日%H:%M:%S", ltm);
  printf(cstr);
  std::cout << std::endl;

  char chm[4];
  strftime(chm, 100, "%H%M", ltm);
  std::string hourmin(chm);
  std::cout << "hourmin: " << hourmin << std::endl;

  return 0;
}
