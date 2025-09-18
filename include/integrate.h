#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "ecs.h"
#include <istream>

void integrate(Body* next_state, Body* current_state, int size,
        float time_delta);

// for simulation purpose, doesn't really have to be that big/small
constexpr float GRAVITATIONAL_CONSTANT = 1.0f;
#endif
