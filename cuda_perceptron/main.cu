#include <iostream>

typedef struct Perceptron {
    float* weights;
    float bias;
    unsigned num_weights;
} Perceptron_t;

__device__ float sigmoid(const float input) {
    return 1.0f / (1.0f + expf(-input));
}

__device__ float loss_function(const float prediction, const float target) {
    return prediction - target;
}

__device__ float summation_function(const float* inputs, const Perceptron_t* perceptron) {
    float sum = perceptron->bias;
    for (unsigned i = 0; i < perceptron->num_weights; ++i) {
        sum += inputs[i] * perceptron->weights[i];
    }
    return sum;
}

__device__ void update_weights(Perceptron_t* perceptron, const float* inputs, const float error, const float learning_rate) {
    for (unsigned i = 0; i < perceptron->num_weights; ++i) {
        perceptron->weights[i] -= learning_rate * error * inputs[i];
    }
    perceptron->bias -= learning_rate * error;
}

__global__ void train(Perceptron_t* perceptron, const float* inputs, const float* targets, const float learning_rate) {
    const float sum = summation_function(inputs, perceptron);
    const float prediction = sigmoid(sum);
    const float error = prediction - *targets;
    update_weights(perceptron, inputs, error, learning_rate);
}

int main() {
    constexpr unsigned inputs = 4;
    constexpr float learning_rate = 0.5f;

    constexpr float h_inputs[inputs] = {1.0f, 0.0f, 1.0f, 0.0f};
    constexpr float h_target = 1.0f;
    constexpr float h_weights[2] = {0.1f, 0.2f};
    constexpr float h_bias = 0.5f;

    float *d_inputs, *d_target, *d_weights;
    Perceptron_t *d_perceptron;

    cudaMalloc(&d_inputs, inputs * sizeof(float));
    cudaMalloc(&d_target, sizeof(float));
    cudaMalloc(&d_weights, 2 * sizeof(float));
    cudaMalloc(&d_perceptron, sizeof(Perceptron_t));

    cudaMemcpy(d_inputs, h_inputs, inputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &h_target, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, 2 * sizeof(float), cudaMemcpyHostToDevice);

    const Perceptron_t h_perceptron = {d_weights, h_bias, 2};
    cudaMemcpy(d_perceptron, &h_perceptron, sizeof(Perceptron_t), cudaMemcpyHostToDevice);

    train<<<1, 1>>>(d_perceptron, d_inputs, d_target, learning_rate);
    cudaDeviceSynchronize();

    cudaFree(d_inputs);
    cudaFree(d_target);
    cudaFree(d_weights);
    cudaFree(d_perceptron);

    return 0;
}
