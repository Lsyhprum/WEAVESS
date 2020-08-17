//
// Created by Murph on 2020/8/15.
//

#include "weavess/distance.h"
#include <cassert>
//#include <boost/test>

template<typename T>
static T ComputeL2Distance(const T *pX, const T *pY, int length) {
    float diff = 0;
    const T* pEnd1 = pX + length;
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
    }
    return diff;
}

template<typename T>
T random(int high = RAND_MAX, int low = 0)   // Generates a random value.
{
    return (T)(low + float(high - low)*(std::rand()/static_cast<float>(RAND_MAX + 1.0)));
}

template <typename T>
void test(int high, int dim){
    T *X = new T[dim], *Y = new T[dim];
    //BOOST_ASSERT(X != nullptr && Y != nullptr);

    for(int i = 0; i < dim; i ++){
        X[i] = random<T>(high, -high);
        Y[i] = random<T>(high, -high);
    }

    weavess::Distance dis;

    for(int i = 0; i < dim; i ++){
        std::cout << X[i] << std::endl;
    }
    std::cout << dis.compare(X, Y, dim) << std::endl;

    BOOST_CHECK_CLOSE_FRACTION(dis.compare<float>(X, Y, dim), ComputeL2Distance<float>(X, Y, dim), 1e-5f);

    delete[] X;
    delete[] Y;
}

//BOOST_AUTO_TEST_SUITE(DistanceTest)
//
//BOOST_AUTO_TEST_CASE(){
//    test<float>(1);
//    test<std::int8_t>(127);
//    test<std::int16_t>(32767);
//}
//
//BOOST_AUTO_TEST_SUITE_END()
//
//int main(){
//    test<float>(100, 10);
//    return 0;
//}