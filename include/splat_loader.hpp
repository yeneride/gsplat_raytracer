#pragma once
#include <string>
#include <vector>
#include "gaussian.hpp"

class SplatLoader {
public:
  static std::vector<Gaussian> load_from_ply(const std::string &filename);
}; 
