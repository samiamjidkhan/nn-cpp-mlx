#include <iostream>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  auto W = mx::array({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {3, 2});
  auto x = mx::array({1.0, 2.0}, {2});
  auto V = mx::array({0.7, 0.8, 0.9}, {1, 3});
  auto t = mx::array({2.0f}, {1});
  
  float lr = 0.1f;
  int steps = 20;
  
  for (int i = 0; i < steps; ++i) {
    // Forward pass
    auto z = mx::matmul(W, x);
    auto a = mx::maximum(z, mx::array(0.0f));  // ReLU
    auto y = mx::matmul(V, a);
    
    // Loss: 0.5 * (y - t)^2
    auto diff = y - t;
    auto loss = 0.5f * mx::sum(diff * diff);
    
    if (i % 2 == 0 || i == steps - 1) {
      std::cout << "step=" << i << ", loss=" << loss << ", y=" << y << std::endl;
    }
    
    // Backward pass
    auto dy = diff;                                              // dL/dy
    auto dV = mx::outer(dy, a);                                  // (1,3)
    auto dA = mx::reshape(mx::matmul(mx::transpose(V), dy), {3}); // (3)
    auto dZ = mx::where(mx::greater(z, mx::array(0.0f)), dA, mx::array(0.0f)); // ReLU gradient
    auto dW = mx::outer(dZ, x);                                  // (3,2)
    
    // Update weights
    W = W - lr * dW;
    V = V - lr * dV;
  }
  
  return 0;
}
