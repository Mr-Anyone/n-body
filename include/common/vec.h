#ifndef MATH_VEC_H 
#define MATH_VEC_H 

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#include <random>
#include <cmath>

struct Vec2{
    float x, y;
    inline HD Vec2()
        :x(0.0), y(0.0){
    }

    inline HD Vec2(float x, float y)
        :x(x), y(y){
    }

    HD inline Vec2 operator+(const Vec2& rhs) const{
        Vec2 result; 
        result.x = x + rhs.x;
        result.y = y + rhs.y;
        return result;
    }

    HD inline Vec2 operator-(const Vec2& rhs) const{
        Vec2 result; 
        result.x = x - rhs.x;
        result.y = y - rhs.y;
        return result;
    }

    HD inline float length() const{
        return std::sqrt(x*x + y*y);
    }

    HD inline Vec2 operator*(float constant){
        Vec2 result; 
        result.x = x*constant;
        result.y = y*constant;
        return result;
    }
};
inline float randFloat(){
    static std::random_device rd;
    static std::mt19937 gen(rd());  // Mersenne Twister engine
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(gen);
}


inline Vec2 getRandVec(){
    return Vec2(randFloat(), randFloat());
}
#endif
