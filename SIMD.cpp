/*
 * Copyright 2022 BDAP team.
 *
 * Author: Laurens Devos
 * Version: 0.1
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <cmath>
#include <arm_neon.h>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

/**
 * A matrix representation.
 *
 * Based on:
 * https://github.com/laudv/veritas/blob/main/src/cpp/basics.hpp#L39
 */
template <typename T>
struct matrix {
private:
    std::vector<T> vec_;

public:
    size_t nrows, ncols;
    size_t stride_row, stride_col; // in num of elems, not bytes

    /** Compute the index of an element. */
    inline size_t index(size_t row, size_t col) const
    {
        if (row >= nrows)
            throw std::out_of_range("out of bounds row");
        if (col >= ncols)
            throw std::out_of_range("out of bounds column");
        return row * stride_row + col * stride_col;
    }

    /** Get a pointer to the data */
    inline const T *ptr() const { return vec_.data(); }

    /** Get a pointer to an element */
    inline const T *ptr(size_t row, size_t col) const
    { return &ptr()[index(row, col)]; }

    /** Get a pointer to the data */
    inline T *ptr_mut() { return vec_.data(); }

    /** Get a pointer to an element */
    inline T *ptr_mut(size_t row, size_t col)
    { return &ptr_mut()[index(row, col)]; }

    /** Access element in data matrix without bounds checking. */
    inline T get_elem(size_t row, size_t col) const
    { return ptr()[index(row, col)]; }

    
    /** Access element in data matrix without bounds checking. */
    inline void set_elem(size_t row, size_t col, T&& value)
    { ptr_mut()[index(row, col)] = std::move(value); }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](size_t i) const
    { return ptr()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T& operator[](size_t i)
    { return ptr_mut()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](std::pair<size_t, size_t> p) const
    { auto &&[i, j] = p; return get_elem(i, j); }

    matrix(std::vector<T>&& vec, size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(std::move(vec))
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}

    matrix(size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(nr * nc)
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}
};

using fmatrix = matrix<float>;

std::tuple<fmatrix, fmatrix, fmatrix, float>
read_bin_data(const char *fname)
{
    std::ifstream f(fname, std::ios::binary);

    char buf[8];
    f.read(buf, 8);

    int num_ex = *reinterpret_cast<int *>(&buf[0]);
    int num_feat = *reinterpret_cast<int *>(&buf[4]);

    std::cout << "num_ex " << num_ex << ", num_feat " << num_feat << std::endl;

    size_t num_numbers = num_ex * num_feat;
    fmatrix x(num_ex, num_feat, num_feat, 1);
    fmatrix y(num_ex, 1, 1, 1);
    fmatrix coef(num_feat, 1, 1, 1);

    f.read(reinterpret_cast<char *>(x.ptr_mut()), num_numbers * sizeof(float));
    f.read(reinterpret_cast<char *>(y.ptr_mut()), num_ex * sizeof(float));
    f.read(reinterpret_cast<char *>(coef.ptr_mut()), num_feat * sizeof(float));

    f.read(buf, sizeof(float));
    float intercept = *reinterpret_cast<float *>(&buf[0]);

    return std::make_tuple(x, y, coef, intercept);
}

// Grading: most optimal accurate SIMD implementation I could achieve
fmatrix evaluate_simd(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);

    // Define the chunks of the matrix to unroll the loop
    constexpr auto FLOATS_IN_REGISTER = 4;
    const auto vectorizableSamples = (x.ncols / FLOATS_IN_REGISTER) * FLOATS_IN_REGISTER;
    
    // Go row by row
    for (size_t i = 0; i < x.nrows; i ++) {
        
        // Process 4 columns at a time by loading them into the registers as vectors
        size_t j = 0;
        output[i] = 0;
        for (; j < vectorizableSamples; j += 4) {
            float32x4_t register1 = vld1q_f32(x.ptr(i, j));

            float32x4_t register2 = vld1q_f32(coef.ptr(j,0));

            float32_t final = vaddvq_f32(vmulq_f32(register1, register2));

            output[i] += final;
        } 

        // Continue with the remaining individual columns scalarly
        for (; j < x.ncols; j++) {
            output[i] += x.get_elem(i,j) * coef.get_elem(j,0);
        }

        output[i] += intercept;
        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));
    }
    return output;
}

// Grading: most optimal accurate scalar implementation I could achieve
fmatrix evaluate_scalar(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{   
    fmatrix output(x.nrows, 1, 1, 1);

    // Define the chunks of the matrix to unroll the loop
    const auto maxUnrollingEight = (x.ncols / 8) * 8;
    const auto maxUnrollingFour = (x.ncols / 4) * 4;

    // Iterate row per row
    for (size_t i = 0; i < x.nrows; i++){ 
        output[i] = 0;
        // Try unrolling column loop in strides of 8
        size_t j = 0;
        for (; j < maxUnrollingEight; j += 8) {
            for (size_t k = 0; k < 8; k++) {
                output[i] += x.get_elem(i, j + k) * coef.get_elem(j + k, 0);
            }
        }

        // If that fails, try unrolling the remaining column loop in strides of 4
        for (; j < maxUnrollingFour; j += 4) {
            for (size_t k = 0; k < 4; k++) {
                output[i] += x.get_elem(i, j + k) * coef.get_elem(j + k, 0);
            }
        }

        // Continue with the remaining columns individually
        for (; j < x.ncols; j ++) {
            output[i] += x.get_elem(i,j) * coef.get_elem(j,0);
        }

        // Finally, add the intercept
        output[i] += intercept;
        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));

    }
    return output;
}

// Benchmarking: naive implementation to compare performance
fmatrix evaluate_scalar_naive(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{   
    fmatrix output(x.nrows, 1, 1, 1);
    for (size_t i = 0; i < x.nrows; i++){ 
        for (size_t j = 0; j < x.ncols; j++) {
            output[i] += x.get_elem(i,j) * coef.get_elem(j,0);
        }
        output[i] += intercept;
        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));
    }
    return output;
}

// Demonstration: Attempt to increase SIMD performance; faster but less accurate
fmatrix evaluate_simd_16(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);
    constexpr auto FLOATS_IN_REGISTER = 4;
    const auto vectorizableSamplesRows = (x.nrows / 16) * 16;
    const auto vectorizableSamples = (x.ncols / FLOATS_IN_REGISTER) * FLOATS_IN_REGISTER;
    
    size_t i = 0;
    for (; i < vectorizableSamplesRows; i += 16) {
        output[i] = 0;
        output[i+1] = 0;
        output[i+2] = 0;
        output[i+3] = 0;
        output[i+4] = 0;
        output[i+5] = 0;
        output[i+6] = 0;
        output[i+7] = 0;
        output[i+8] = 0;
        output[i+9] = 0;
        output[i+10] = 0;
        output[i+11] = 0;
        output[i+12] = 0;
        output[i+13] = 0;
        output[i+14] = 0;
        output[i+15] = 0;
        size_t j = 0;
        for (; j < vectorizableSamples; j += FLOATS_IN_REGISTER) {
            float32x4_t registerC = vld1q_f32(coef.ptr()+j); 

            float32x4_t register1 = vld1q_f32(x.ptr(i, j));
            float32x4_t register2 = vld1q_f32(x.ptr(i + 1, j));
            float32x4_t register3 = vld1q_f32(x.ptr(i + 2, j));
            float32x4_t register4 = vld1q_f32(x.ptr(i + 3, j));
            float32x4_t register5 = vld1q_f32(x.ptr(i + 4, j));
            float32x4_t register6 = vld1q_f32(x.ptr(i + 5, j));
            float32x4_t register7 = vld1q_f32(x.ptr(i + 6, j));
            float32x4_t register8 = vld1q_f32(x.ptr(i + 7, j));
            float32x4_t register9 = vld1q_f32(x.ptr(i + 8, j));
            float32x4_t register10 = vld1q_f32(x.ptr(i + 9, j));
            float32x4_t register11 = vld1q_f32(x.ptr(i + 10, j));
            float32x4_t register12 = vld1q_f32(x.ptr(i + 11, j));
            float32x4_t register13 = vld1q_f32(x.ptr(i + 12, j));
            float32x4_t register14 = vld1q_f32(x.ptr(i + 13, j));
            float32x4_t register15 = vld1q_f32(x.ptr(i + 14, j));
            float32x4_t register16 = vld1q_f32(x.ptr(i + 15, j));

            float32_t final1 = vaddvq_f32(vmulq_f32(register1, registerC));
            float32_t final2 = vaddvq_f32(vmulq_f32(register2, registerC));
            float32_t final3 = vaddvq_f32(vmulq_f32(register3, registerC));
            float32_t final4 = vaddvq_f32(vmulq_f32(register4, registerC));
            float32_t final5 = vaddvq_f32(vmulq_f32(register5, registerC));
            float32_t final6 = vaddvq_f32(vmulq_f32(register6, registerC));
            float32_t final7 = vaddvq_f32(vmulq_f32(register7, registerC));
            float32_t final8 = vaddvq_f32(vmulq_f32(register8, registerC));
            float32_t final9 = vaddvq_f32(vmulq_f32(register9, registerC));
            float32_t final10 = vaddvq_f32(vmulq_f32(register10, registerC));
            float32_t final11 = vaddvq_f32(vmulq_f32(register11, registerC));
            float32_t final12 = vaddvq_f32(vmulq_f32(register12, registerC));
            float32_t final13 = vaddvq_f32(vmulq_f32(register13, registerC));
            float32_t final14 = vaddvq_f32(vmulq_f32(register14, registerC));
            float32_t final15 = vaddvq_f32(vmulq_f32(register15, registerC));
            float32_t final16 = vaddvq_f32(vmulq_f32(register16, registerC));

            output[i] += final1;
            output[i+1] += final2;
            output[i+2] += final3;
            output[i+3] += final4;
            output[i+4] += final4;
            output[i+5] += final6;
            output[i+6] += final7;
            output[i+7] += final8;
            output[i+8] += final9;
            output[i+9] += final10;
            output[i+10] += final11;
            output[i+11] += final12;
            output[i+12] += final13;
            output[i+13] += final14;
            output[i+14] += final15;
            output[i+15] += final16;
        } 

        for (; j < x.ncols; j++) {
            float theCoef = coef.get_elem(j,0);
            output[i] += x.get_elem(i,j) * theCoef;
            output[i+1] += x.get_elem(i+1,j) * theCoef;
            output[i+2] += x.get_elem(i+2,j) * theCoef;
            output[i+3] += x.get_elem(i+3,j) * theCoef;
            output[i+4] += x.get_elem(i+4,j) * theCoef;
            output[i+5] += x.get_elem(i+5,j) * theCoef;
            output[i+6] += x.get_elem(i+6,j) * theCoef;
            output[i+7] += x.get_elem(i+7,j) * theCoef;
            output[i+8] += x.get_elem(i+8,j) * theCoef;
            output[i+9] += x.get_elem(i+9,j) * theCoef;
            output[i+10] += x.get_elem(i+10,j) * theCoef;
            output[i+11] += x.get_elem(i+11,j) * theCoef;
            output[i+12] += x.get_elem(i+12,j) * theCoef;
            output[i+13] += x.get_elem(i+13,j) * theCoef;
            output[i+14] += x.get_elem(i+14,j) * theCoef;
            output[i+15] += x.get_elem(i+15,j) * theCoef;
        }

        output[i] += intercept;
        output[i+1] += intercept;
        output[i+2] += intercept;
        output[i+3] += intercept;
        output[i+4] += intercept;
        output[i+5] += intercept;
        output[i+6] += intercept;
        output[i+7] += intercept;
        output[i+8] += intercept;
        output[i+9] += intercept;
        output[i+10] += intercept;
        output[i+11] += intercept;
        output[i+12] += intercept;
        output[i+13] += intercept;
        output[i+14] += intercept;
        output[i+15] += intercept;

        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));     
        //output[i+1] = 1.0 / (1.0 + exp(-(output[i+1])));     
        //output[i+2] = 1.0 / (1.0 + exp(-(output[i+2])));     
        //output[i+3] = 1.0 / (1.0 + exp(-(output[i+3])));     
        //output[i+4] = 1.0 / (1.0 + exp(-(output[i+4])));     
        //output[i+5] = 1.0 / (1.0 + exp(-(output[i+5])));     
        //output[i+6] = 1.0 / (1.0 + exp(-(output[i+6])));     
        //output[i+7] = 1.0 / (1.0 + exp(-(output[i+7])));     
        //output[i+8] = 1.0 / (1.0 + exp(-(output[i+8])));     
        //output[i+9] = 1.0 / (1.0 + exp(-(output[i+9])));     
        //output[i+10] = 1.0 / (1.0 + exp(-(output[i+10])));     
        //output[i+11] = 1.0 / (1.0 + exp(-(output[i+11])));     
        //output[i+12] = 1.0 / (1.0 + exp(-(output[i+12])));     
        //output[i+13] = 1.0 / (1.0 + exp(-(output[i+13])));     
        //output[i+14] = 1.0 / (1.0 + exp(-(output[i+14])));     
        //output[i+15] = 1.0 / (1.0 + exp(-(output[i+15])));                   
    }

    for (; i < x.nrows; i ++) {
        
        size_t j = 0;
        output[i] = 0;
        for (; j < vectorizableSamples; j += FLOATS_IN_REGISTER) {
            float32x4_t register1 = vld1q_f32(x.ptr(i, j));
            float32x4_t register2 = vld1q_f32(coef.ptr()+j); 

            float32x4_t intermediate1 = vmulq_f32(register1, register2);

            float32_t final = vaddvq_f32(intermediate1);

            output[i] += final;
        } 

        for (; j < x.ncols; j++) {
            output[i] += x.get_elem(i,j) * coef.get_elem(j,0);
        }

        output[i] += intercept;
        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));     
    }
    return output;
}

// Demonstration: SIMD implementation using more vectorization,
// but doesn't yield performance improvements
fmatrix evaluate_simd_wider(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);

    // Define the chunks of the matrix to unroll the loop
    constexpr auto FLOATS_IN_REGISTER = 8;
    const auto vectorizableSamples = (x.ncols / FLOATS_IN_REGISTER) * FLOATS_IN_REGISTER;
    
    // Go row by row
    for (size_t i = 0; i < x.nrows; i ++) {
        
        // Process 8 columns at a time by loading them into the registers as vectors
        size_t j = 0;
        output[i] = 0;
        for (; j < vectorizableSamples; j += 8) {
            float32x4_t registerX1 = vld1q_f32(x.ptr(i, j));
            float32x4_t registerX2 = vld1q_f32(x.ptr(i, j + 4));

            float32x4_t registerC1 = vld1q_f32(coef.ptr(j,0));
            float32x4_t registerC2 = vld1q_f32(coef.ptr(j+4,0)); 

            float32_t final1 = vaddvq_f32(vmulq_f32(registerX1, registerC1));
            float32_t final2 = vaddvq_f32(vmulq_f32(registerX2, registerC2));

            output[i] += final1 + final2;
        } 

        // Continue with the remaining individual columns scalarly
        for (; j < x.ncols; j++) {
            output[i] += x.get_elem(i,j) * coef.get_elem(j,0);
        }

        output[i] += intercept;
        // For logistic regression
        //output[i] = 1.0 / (1.0 + exp(-(output[i])));
    }
    return output;
}

int main(int argc, char *argv[])
{
    // These are four linear regression models
    // 8 features 
    //auto &&[x, y, coef, intercept] = read_bin_data("data/calhouse.bin");
    // 130 features 
    auto &&[x, y, coef, intercept] = read_bin_data("data/allstate.bin"); 
    // 26 features
    //auto &&[x, y, coef, intercept] = read_bin_data("data/diamonds.bin");
    // 12 features
    //auto &&[x, y, coef, intercept] = read_bin_data("data/cpusmall.bin");

    // This is a logistic regression model, but can be evaluated in the same way
    // All you would need to do is apply the sigmoid to the values in `output_*`
    // 784 features: OG scalar, 16 SIMD
    //auto &&[x, y, coef, intercept] = read_bin_data("data/mnist_5vall.bin");
    
    steady_clock::time_point tbegin, tend;
    // Code below was used for local benchmarking

    // BEGINNING OF SPEED EVALUATIONS
    // Evaluation of Scalar in ms
    std::cout << "------ Beginning of evaluation Scalar -------"  << std::endl
        << duration_cast<microseconds>(tend - tbegin).count() 
        << std::endl;

    microseconds sum_scalar_new = microseconds(0);
    for (size_t i = 0; i < 100; ++i) {
        tbegin = steady_clock::now();
        auto output_scalar = evaluate_scalar(x, y, coef, intercept);
        tend = steady_clock::now();
        sum_scalar_new += duration_cast<microseconds>(tend - tbegin);
        std::cout << duration_cast<microseconds>(tend - tbegin).count() / 1000.0 << std::endl;
    }
    auto avg_time_scalar_new = sum_scalar_new / 100.0;

    // Evaluation of Naive Scalar in ms
    std::cout << "------ Beginning of evaluation Naive Scalar -------"  << std::endl
        << duration_cast<microseconds>(tend - tbegin).count() 
        << std::endl;

    microseconds sum_scalar_naive = microseconds(0);
    for (size_t i = 0; i < 100; ++i) {
        tbegin = steady_clock::now();
        auto output_scalar = evaluate_scalar_naive(x, y, coef, intercept);
        tend = steady_clock::now();
        sum_scalar_naive += duration_cast<microseconds>(tend - tbegin);
        std::cout << duration_cast<microseconds>(tend - tbegin).count() / 1000.0 << std::endl;
    }
    auto avg_time_scalar_naive = sum_scalar_naive / 100.0;

    // Evaluation of SIMD in ms
    std::cout << "------ Beginning of evaluation SIMD -------"  << std::endl
        << duration_cast<microseconds>(tend - tbegin).count() 
        << std::endl;

    microseconds sum_simd = microseconds(0);
    for (size_t i = 0; i < 100; ++i) {
        tbegin = steady_clock::now();
        auto output_simd = evaluate_simd(x, y, coef, intercept);
        tend = steady_clock::now();
        sum_simd += duration_cast<microseconds>(tend - tbegin);
        std::cout << duration_cast<microseconds>(tend - tbegin).count() / 1000.0 << std::endl;
    }
    auto avg_time_simd = sum_simd / 100.0;

    // Evaluation of SIMD 16 in ms
    std::cout << "------ Beginning of evaluation SIMD 16 -------"  << std::endl
        << duration_cast<microseconds>(tend - tbegin).count() 
        << std::endl;
        
    microseconds sum_simd_16 = microseconds(0);
    for (size_t i = 0; i < 100; ++i) {
        tbegin = steady_clock::now();
        auto output_simd = evaluate_simd_16(x, y, coef, intercept);
        tend = steady_clock::now();
        sum_simd_16 += duration_cast<microseconds>(tend - tbegin);
        std::cout << duration_cast<microseconds>(tend - tbegin).count() / 1000.0 << std::endl;
    }
    auto avg_time_simd_16 = sum_simd_16 / 100.0;

    std::cout << "------ End of evaluation -------"  << std::endl;

    std::cout << "Evaluated Scalar in an avg time of " << avg_time_scalar_new.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Evaluated Naive Scalar in an avg time of " << avg_time_scalar_naive.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Evaluated SIMD in an avg time of " << avg_time_simd.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Evaluated SIMD 16 in an avg time of " << avg_time_simd_16.count() / 1000.0 << " ms" << std::endl;

    // BEGINNING OF ACCURACY EVALUATIONS
    // calculate predications
    float ys = 0;

    float sumScalar = 0;
    float sqrdErrorScalar = 0;
    auto ysScalar = evaluate_scalar(x, y, coef, intercept);

    float sumScalarNaive = 0;
    float sqrdErrorScalarNaive = 0;
    auto ysScalarNaive = evaluate_scalar_naive(x, y, coef, intercept);

    float sumSIMD = 0;
    float sqrdErrorSIMD = 0;
    auto ysSIMD = evaluate_simd(x, y, coef, intercept);

    float sumSIMD16 = 0;
    float sqrdErrorSIMD16 = 0;
    auto ysSIMD16 = evaluate_simd_16(x, y, coef, intercept);

    // Calculate total sum and squared error
    for (size_t k = 0; k < y.nrows; k++) {
        ys += y[k];

        sumScalar += ysScalar[k];
        sqrdErrorScalar += (y[k] - ysScalar[k]) * (y[k] - ysScalar[k]);

        sumScalarNaive += ysScalarNaive[k];
        sqrdErrorScalarNaive += (y[k] - ysScalarNaive[k]) * (y[k] - ysScalarNaive[k]);

        sumSIMD += ysSIMD[k];
        sqrdErrorSIMD += (y[k] - ysSIMD[k]) * (y[k] - ysSIMD[k]);

        sumSIMD16 += ysSIMD16[k];
        sqrdErrorSIMD16 += (y[k] - ysSIMD16[k]) * (y[k] - ysSIMD16[k]);
    }

    // Output results
    std::cout << "Sum of ys: " << ys << std::endl;
    std::cout << "Squared error Scalar: " << sqrdErrorScalar << std::endl;
    std::cout << "Squared error Scalar Naive: " << sqrdErrorScalarNaive << std::endl;
    std::cout << "Squared error SIMD: " << sqrdErrorSIMD << std::endl;
    std::cout << "Squared error SIMD 16: " << sqrdErrorSIMD16 << std::endl;
    std::cout << "Sum Scalar: " << sumScalar << std::endl;
    std::cout << "Sum Scalar Naive: " << sumScalarNaive << std::endl;
    std::cout << "Sum SIMD: " << sumSIMD << std::endl;
    std::cout << "Sum SIMD 16: " << sumSIMD16 << std::endl;
}