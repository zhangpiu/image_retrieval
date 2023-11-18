#include <iostream>

#include "cmdline/cmdline.h"
#include "eigen3/Eigen/Dense"

using namespace std;

int main(int argc, char* argv[]) {
  // Set stdout unbuffered
  std::setbuf(stdout, nullptr);

  cmdline::parser parser;
  parser.add<std::string>(
      "input", 'i', "Input filename(if empty, read from stdin)", false, "");
  parser.add<std::string>("centroids", 'c', "Output centroids", true);
  parser.add<std::string>("membership", 'm', "Output membership", true);
  parser.add<size_t>("limit", 'l', "Test limit", false,
                     std::numeric_limits<size_t>::max());
  parser.add("help", 0, "print this message");
  bool ok = parser.parse(argc, argv);
  if (not ok) {
    std::cerr << parser.usage();
    return 1;
  }

  // TODO
  return 0;
}
