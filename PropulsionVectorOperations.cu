//
// Created by steve on 7/14/2020.
//
#include "Propulsion.cuh"

template<typename type, typename aType>
int Propulsion::vectorAngleBetweenVectors(type &angle, aType *a, aType *b, unsigned size)
{
    // Angle between vectors is defined via -> a . b
    //                                         ------
    //                                         |a||b|
    type denominatorA = Propulsion::vectorMagnitude<type, aType>(a, size);
    type denominatorB = Propulsion::vectorMagnitude<type, aType>(b, size);
    if(denominatorA == 0 || denominatorB == 0)
    {
        return -1;
    }

    type numerator = Propulsion::vectorDotProduct<type, aType>(a,b,size);
    angle = numerator / (denominatorA * denominatorB);

    return 0;
}


template<typename type, typename aType>
type Propulsion::vectorDotProduct(aType *a, aType *b, unsigned size)
{
    type dotProduct = 0;
    for(unsigned i = 0; i < size; i++)
    {
        dotProduct += a[i]*b[i];
    }
    return dotProduct;
}

template<typename type, typename aType>
type Propulsion::vectorMagnitude(aType *a, unsigned  size)
{
    type magnitude = 0;
    for(unsigned i = 0; i < size; i++)
    {
        magnitude += a[i]*a[i];
    }
    return std::sqrt(magnitude);
}

template<typename type>
type *Propulsion::vectorScalar(type *a, type scalar, unsigned size)
{
    type* r = new type[size];
    for(unsigned i = 0; i < size; i++)
    {
        r[i] = a[i] * scalar;
    }
    return r;
}

template<typename type>
type *Propulsion::vectorAdd(type *a, type *b, unsigned  size)
{
    type *c = new type[size];
    for(unsigned short i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}

template<typename type>
type *Propulsion::vectorSubtract(type *a, type *b, unsigned  size)
{
    type *c = new type[size];
    for(unsigned short i = 0; i < size; i++)
    {
        c[i] = a[i] - b[i];
    }
    return c;
}



template<typename type>
type * Propulsion::vectorCrossProduct(type *a, type *b)
{
    // Cross Product of a vector is described as:
    // AxB = |a||b|sin(o)*n
    // or
    // c_x = a_y * b_z - a_z * b_y
    // c_y = a_z * b_x - a_x * b_z
    // c_z = a_x * b_y - a_y * b_x

    // Create vector for that type.
    type *c = new type[3];

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
    return c;
}
