#include "splat_loader.hpp"
#include "covariance.hpp"
#include "happly.hpp"
#include "spherical_harmonics.hpp"
#include <glm/glm.hpp>
#include <vector>

std::vector<Gaussian> SplatLoader::load_from_ply(const std::string &filename) {
  happly::PLYData plyIn(filename);
  auto x = plyIn.getElement("vertex").getProperty<float>("x");
  auto y = plyIn.getElement("vertex").getProperty<float>("y");
  auto z = plyIn.getElement("vertex").getProperty<float>("z");
  auto sx = plyIn.getElement("vertex").getProperty<float>("scale_0");
  auto sy = plyIn.getElement("vertex").getProperty<float>("scale_1");
  auto sz = plyIn.getElement("vertex").getProperty<float>("scale_2");
  auto qx = plyIn.getElement("vertex").getProperty<float>("rot_0");
  auto qy = plyIn.getElement("vertex").getProperty<float>("rot_1");
  auto qz = plyIn.getElement("vertex").getProperty<float>("rot_2");
  auto qw = plyIn.getElement("vertex").getProperty<float>("rot_3");
  auto opacities = plyIn.getElement("vertex").getProperty<float>("opacity");
  auto f_dc_0 = plyIn.getElement("vertex").getProperty<float>("f_dc_0");
  auto f_dc_1 = plyIn.getElement("vertex").getProperty<float>("f_dc_1");
  auto f_dc_2 = plyIn.getElement("vertex").getProperty<float>("f_dc_2");

  std::vector<Gaussian> gaussians;
  for (int i = 0; i < x.size(); ++i) {
    glm::vec3 mean = glm::vec3(x[i], y[i] + 6.0f, z[i] + 24.0f);
    glm::vec3 scale = glm::vec3(exp(sx[i]), exp(sy[i]), exp(sz[i]));
    glm::vec4 rotation = glm::normalize(glm::vec4(qx[i], qy[i], qz[i], qw[i]));
    float opacity = 1.0 / (1.0 + std::exp(-opacities[i]));
    glm::vec3 coeffs[3] = {glm::vec3(f_dc_0[i], f_dc_1[i], f_dc_2[i]),
                           glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 0.0f, 0.0f)};
    SphericalHarmonics sh(coeffs[0], coeffs[1], coeffs[2]);
    Covariance cov(scale, rotation);
    Gaussian g(mean, cov, sh, opacity);
    gaussians.push_back(g);
  }
  return gaussians;
}
