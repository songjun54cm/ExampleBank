#include <iostream>
#include <time.h>
#include <string>

int test1() {
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

void GetCurrentHHmmTime(char* char_hour_min ) {
  time_t current_time = time(NULL);
  struct tm now_time;
  localtime_r(&current_time, &now_time);
  strftime(char_hour_min, 100, "%H%M", &now_time);
}

int test2() {
  char char_hour_min[4];
  GetCurrentHHmmTime(char_hour_min);
  std::string current_hour_minute(char_hour_min);
  std::cout << "current HHmm: " << current_hour_minute;

  return 0;
}

int main() {
  return test2();
}
