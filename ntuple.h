#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <functional>
#include <fstream>

template<int DIMS, int DIM_SIZE, typename T>
class NTupleTable {
public:
    using NTuple = std::array<unsigned char, DIMS>;

private:
    std::vector<float> data;

    int tuple_to_index(const NTuple &tuple) const {
        int address = 0;
        for (int i = 0; i < DIMS; i++) {
            address += (tuple[i] * std::pow(DIM_SIZE, ((DIMS - 1) - i)));
        }
        return address;
    }

public:

    NTupleTable() : data(std::llround(std::pow(DIM_SIZE, DIMS)), 0) {}

    float& operator()(const NTuple &tuple) {
        return data[tuple_to_index(tuple)];
    }

    const float& operator()(const NTuple &tuple) const {
        return data[tuple_to_index(tuple)];
    }

    void save(std::ofstream &stream) const {
        stream.write((const char*) &data[0], (data.size() * sizeof(float)));
    }

    void load(std::ifstream &stream) {
        stream.read((char*) &data[0], (data.size() * sizeof(float)));
    }
};