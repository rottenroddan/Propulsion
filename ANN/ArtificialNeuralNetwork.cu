//
// Created by steve on 8/10/2020.
//

#include "Propulsion.cuh"

void Propulsion::ArtificialNeuralNetwork::test()
{
    // bullshit input for testing....
    /*
    double input_Arr[] = {1.0,2.0,3.0,2.5,
                          2.0,5.0,-1.0,2.0,
                          -1.5,2.7,3.3,-0.8};*/

    double input_Arr[] =
    { 0.0, 0.0, 1.0,
      1.0, 1.0, 1.0,
      1.0, 0.0, 1.0,
      0.0, 1.0, 1.0
    };

    double y[] = {0.0,1.0,1.0,0.0};


    // input stored as a matrix.
    Matrix<double> input(input_Arr,4,3);
    Matrix<double> output(y,1,4);

    // first layer ex:
    auto firstLayer = LayerDense(3,1);
    //auto secondLayer = LayerDense(10,256);
    //auto thirdLayer = LayerDense(256,1);

    firstLayer.forward(input);



    auto activationLayer = ActivationSigmoid();
    activationLayer.forward(firstLayer);
    activationLayer.printOutputLayer();




    auto softmax = ActivationSoftmax();
    softmax.forward(firstLayer);
}