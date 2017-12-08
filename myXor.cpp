#include "myXor.hpp"

void std_vec_xor(uint8_t * v1, uint8_t * v2, int num_el) {
  /*
  This function XORs two vectors v1 and v2, which are specified through their
  begin position and end position. The result overwrites v1.
  */
  // Determine the number of bytes to XOR.
  /* xor two std::vectors to the end of the shorter vector */
  // XOR the bytes.
  for (int ii = 0; ii < num_el; ii++) {
    *v1++ ^= *v2++;
  }
}


void std_vec_minus(float * v1, float * v2, float factor, int num_el) {
  for (int ii = 0; ii < num_el; ii++) {
    *v1++ -= (*v2++) * factor;
  }
}

int main(){
  uint8_t x[] = {0x11, 0x22};
  uint8_t y[] = {0x12, 0x21};
  std_vec_xor(x, y, 2);
  std::cout << int(x[0]) << ' ' << int(x[1]) << std::endl;
  float m[] = {123, 234};
  float n[] = {122, 234};
  std_vec_minus(m, n, 3, 2);
  std::cout << int(m[0]) << ' ' << int(m[1]) << std::endl;
  return 0;
}
