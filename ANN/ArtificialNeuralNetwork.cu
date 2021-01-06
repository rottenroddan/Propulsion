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
    /*
    double input_Arr[] =
    { 0.0, 0.0, 1.0,
      1.0, 1.0, 1.0,
      1.0, 0.0, 1.0,
      0.0, 1.0, 1.0,
      0.0, 0.0, 0.0
    };

    double y[] = {0.0,1.0,1.0,0.0};*/

    double input_Arr[] =
            { 0.0, 0.0, 1.0,
              1.0, 1.0, 1.0,
              1.0, 0.0, 1.0,
              0.0, 1.0, 1.0,
              0.0, 0.0, 0.0
            };

    double y[] = {1.0,0.0,0.0,
                  0.0,1.0,0.0,
                  0.0,1.0,0.0,
                  1.0,0.0,0.0,
                  0.0,0.0,1.0};


    // input stored as a matrix.
    Matrix<double> input(input_Arr,5,3);
    std::shared_ptr<Matrix<double>> y_true(new Propulsion::Matrix<double>(y,5,3));

    // first layer ex:
    auto firstLayer = LayerDense(3,10);
    auto secondLayer = LayerDense(10,256);
    auto thirdLayer = LayerDense(256,3);

    firstLayer.forward(input);
    std::cout << "First Layer with Input" << std::endl;
    firstLayer.printOutputLayer();
    std::cout << "----------------------" << std::endl;


    secondLayer.forward(firstLayer);
    thirdLayer.forward(secondLayer);






    auto softmax = ActivationSoftmax();
    softmax.forward(thirdLayer);
    softmax.getOutputLayer()->print();

    auto loss_function = LossCategoricalCrossentropy();
    loss_function.calculate(softmax.getOutputLayer(), y_true );
}