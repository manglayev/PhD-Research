#include "SIC_header.cuh"
class SIC_Random
{
  public:
    float uniform0to1Random()
    {
        float r = random();
        return r / ((float)RAND_MAX + 1);
    }
    float getRandomFloat()
    {
      return 2.0 * uniform0to1Random() - 1.0;
    }
};
