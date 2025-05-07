#include "image_io.hpp"
#include <algorithm>
#include <fstream>

void write_ppm(const char *filename, const float *img, int width, int height) {
  std::ofstream ofs(filename);
  ofs << "P3\n" << width << " " << height << "\n255\n";
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = 3 * (y * width + x);
      int r = (unsigned int)(std::min(std::max(img[idx + 0], 0.0f), 1.0f) * 255.0f);
      int g = (unsigned int)(std::min(std::max(img[idx + 1], 0.0f), 1.0f) * 255.0f);
      int b = (unsigned int)(std::min(std::max(img[idx + 2], 0.0f), 1.0f) * 255.0f);
      ofs << r << " " << g << " " << b << " ";
    }
    ofs << "\n";
  }
  ofs.close();
}
