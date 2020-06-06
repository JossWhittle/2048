#pragma once

#include <cassert>
#include <vector>
#include <array>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <cmath>

template<typename T>
class NTupleValueEstimator {
private:
    T value, sum_error, sum_abs_error;

public:

    NTupleValueEstimator() : value(0), sum_error(0), sum_abs_error(0) {}

    operator const T&() const {
        return value;
    }

    void update(const T &error, const T &beta) {
        const T alpha = (sum_abs_error > 0) ? (std::abs(sum_error) / sum_abs_error) : 1.f;
        value         = std::max(0.f, value + (error * beta * alpha));
        sum_error     += error;
        sum_abs_error += std::abs(error);
    }
};

template<int DIMS, int DIM_SIZE>
class NTupleTable {
public:
    using NTuple      = std::array<unsigned char, DIMS>;
//    using NTupleValue = float;
    using NTupleValue = NTupleValueEstimator<float>;

private:
    std::vector<NTupleValue> data;

    int tuple_to_index(const NTuple &tuple) const {
        int address = 0, offset = 1;
        for (int i = 0; i < DIMS; i++) {
            address += ((int) tuple[i] * offset);
            offset  *= DIM_SIZE;
        }

        if (address >= data.size()) {
            std::cout << "tuple ( ";
            for (int i = 0; i < DIMS; i++) {
                std::cout << (int) tuple[i] << ' ';
            }
            std::cout << ") address " << address << " size " << data.size()
                      << " diff " << ((long long) data.size() - address) << std::endl;
            return 0;
        }

        if( data.size() > 0 ) std::exit(-1);

        return address;
    }

public:

    NTupleTable() : data(std::llround(std::pow(DIM_SIZE, DIMS))) {}

    ~NTupleTable() {
        std::cout << "NTupleTable DESTRUCTOR CALLED" << std::endl;
    }

    NTupleTable(const NTupleTable &other) : data(other.data) {
        std::cout << "NTupleTable COPY CONSTRUCTOR CALLED" << std::endl;
    }

    NTupleTable& operator=(const NTupleTable &other) {
        if (&other == this) return *this;
        data = other.data;
        std::cout << "NTupleTable COPY ASSIGNMENT CALLED" << std::endl;
    }

    NTupleTable(NTupleTable &&other) : data(std::move(other.data)) {
        std::cout << "NTupleTable MOVE CONSTRUCTOR CALLED" << std::endl;
    }

    NTupleTable& operator=(NTupleTable &&other) {
        if (&other == this) return *this;
        data = std::move(other.data);
        std::cout << "NTupleTable MOVE CONSTRUCTOR CALLED" << std::endl;
    }

    int size() const {
        return data.size();
    }

    NTupleValue& operator()(const NTuple &tuple) {
        return data[tuple_to_index(tuple)];
    }

    const NTupleValue& operator()(const NTuple &tuple) const {
        return data[tuple_to_index(tuple)];
    }

    void save(std::ofstream &stream) const {
        stream.write((const char*) &data[0], (data.size() * sizeof(float)));
    }

    void load(std::ifstream &stream) {
        stream.read((char*) &data[0], (data.size() * sizeof(float)));
    }
};