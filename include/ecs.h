#ifndef ECS_H 
#define ECS_H

#include "common/vec.h"

struct Body{
    // force and everything else would be calculated
    // in cuda
    Vec2 pos, vel, acc;
    float mass;

};

#endif
