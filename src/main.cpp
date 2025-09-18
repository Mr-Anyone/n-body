#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector> 

#include "ecs.h"
#include "common/vec.h"
#include "integrate.h"


class BodyManager{
public:
    void emplace(Body&& body){
        buffer_a.emplace_back(body);
        buffer_b.emplace_back(body);
    }

    Body* getCurrent(){
        if(front_buffer_kind == A_BUFFER)
            return buffer_a.data();
        assert(front_buffer_kind == B_BUFFER);
        return buffer_b.data();
    }

    Body* getNext(){
        if(front_buffer_kind == A_BUFFER)
            return buffer_b.data();
        assert(front_buffer_kind == B_BUFFER);
        return buffer_a.data();
    }

    void swap(){
        front_buffer_kind = (front_buffer_kind ==  A_BUFFER) ?
            B_BUFFER : A_BUFFER;
    }
private:
    /// keeping track which buffer we are swapping
    enum Type{
        A_BUFFER, 
        B_BUFFER
    };

    Type front_buffer_kind = A_BUFFER;
    std::vector<Body> buffer_a, buffer_b;
    Body* current_state, *next_state;
};

int main(){
    int planet_count;
    BodyManager manager;
    std::cout << "planet count: ";
    std::cin >> planet_count; 

    // move this to cuda?
    for(int i = 0;i<planet_count; ++i){
        Body body (getRandVec(), getRandVec()
                , getRandVec(), randFloat()); 
        manager.emplace(std::move(body));
    }

    while(true){
        Body* current_state = manager.getCurrent();
        Body* next_state = manager.getNext(); 

        integrate(next_state, current_state, 512, 0.01);
        manager.swap(); // the next_state now have the current state 
    }

    return 0;
}
