#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#ifndef VERSION
#define VERSION <QproDCUTemplate_v1.hpp>
#endif // VERSION

#include VERSION

using namespace std;

int main(int argc, char **argv) {
     float gflops = QproDCUTempalte();
     cout << "gflops: " << gflops << endl;
     return 0;
}
