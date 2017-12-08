#include <vector>
#include <iostream>
#include <stdint.h>
#include <math.h>
#include <random>
#include <ctime>
#include <typeinfo>
#include <functional>
#include <numeric>
typedef std::vector<uint8_t>::iterator vec_iter;
typedef std::vector<float>::iterator vec_iter_float;

void std_vec_xor(uint8_t *, uint8_t *, int);
void std_vec_minus(float *, float *, float, int);
