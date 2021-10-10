//
// Copyright 2021 Intel (Autonomous Agents Lab)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include <cassert>
#include <iterator>
#include <vector>

#include "asr_config.h"

ASR_NAMESPACE_BEGIN

template <class T>
struct NestedVector {
    void insert()  // insert empty sub vector
    {
        prefix_sum.push_back(0 + (prefix_sum.empty() ? 0 : prefix_sum.back()));
    }

    void insert(const T& value) {
        prefix_sum.push_back(1 + (prefix_sum.empty() ? 0 : prefix_sum.back()));
        data.push_back(value);
    }

    template <class Iter>
    void insert(const Iter& begin, const Iter& end) {
        ssize_t n_values = std::distance(begin, end);
        assert(n_values >= 0);
        prefix_sum.push_back(n_values +
                             (prefix_sum.empty() ? 0 : prefix_sum.back()));
        data.insert(data.end(), begin, end);
    }

    void insert(const NestedVector<T>& other) {
        for (int i = 0; i < other.size(); ++i) {
            auto r = other.range(i);
            insert(r.first, r.second);
        }
    }

    size_t count(size_t i) const {
        return prefix_sum[i] - (i > 0 ? prefix_sum[i - 1] : 0);
    }

    size_t size() const { return prefix_sum.size(); }

    std::pair<typename std::vector<T>::const_iterator,
              typename std::vector<T>::const_iterator>
    range(size_t i) const {
        typename std::vector<T>::const_iterator begin =
                data.begin() + (i == 0 ? 0 : prefix_sum[i - 1]);
        typename std::vector<T>::const_iterator end = begin + count(i);
        return std::make_pair(begin, end);
    }

    std::pair<typename std::vector<T>::iterator,
              typename std::vector<T>::iterator>
    range(size_t i) {
        typename std::vector<T>::iterator begin =
                data.begin() + (i == 0 ? 0 : prefix_sum[i - 1]);
        typename std::vector<T>::iterator end = begin + count(i);
        return std::make_pair(begin, end);
    }

    T* getPtr(size_t i) { return &data[(i == 0 ? 0 : prefix_sum[i - 1])]; }

    size_t getIdx(size_t i) { return (i == 0 ? 0 : prefix_sum[i - 1]); }

    void clear() {
        prefix_sum.clear();
        data.clear();
    }

    std::vector<size_t> prefix_sum;  // inclusive prefix sum
    std::vector<T> data;
};

ASR_NAMESPACE_END