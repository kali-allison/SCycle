#include <boost/circular_buffer.hpp>
#include <iostream>
#include <cstdio>

int main() {
  boost::circular_buffer<double> _errA;
  _errA.resize(2);
  _errA.push_front(1);
  _errA.push_front(2);
  printf("_errA[0] = %.2e\n", _errA[0]);
  printf("_errA[1] = %.2e\n", _errA[1]);

  return 0;
}
  
