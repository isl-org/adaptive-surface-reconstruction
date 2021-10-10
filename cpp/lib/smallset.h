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

#include <algorithm>
#include <array>
#include <exception>
#include <initializer_list>

#include "asr_config.h"

ASR_NAMESPACE_BEGIN

template <class T, int N>
class SmallSet {
public:
    SmallSet() : pos(0) {}

    SmallSet(std::initializer_list<T> init) {
        pos = 0;
        for (const auto& value : init) {
            insert(value);
        }
    }

    template <class InputIt>
    SmallSet(InputIt first, InputIt last) {
        pos = 0;
        for (auto it = first; it != last; ++it) insert(*it);
    }

    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        for (auto it = first; it != last; ++it) insert(*it);
    }

    void insert(const T& value) {
        auto end_ = data.begin() + pos;
        auto it = std::lower_bound(data.begin(), end_, value);
        if (it == end_ || *it != value) {
            ++pos;
            if (pos > N) {
                --pos;
                throw std::runtime_error(
                        "Number of elements exceeds the capacity of the "
                        "SmallSet");
            }

            auto new_end = data.begin() + pos;

            T tmp;
            tmp = *it;
            *it = value;
            for (auto it2 = it + 1; it2 != new_end; ++it2) {
                std::swap(tmp, *it2);
            }
        }
    }

    int count(const T& value) const {
        auto end_ = data.begin() + pos;
        auto it = std::lower_bound(data.begin(), end_, value);
        if (it == end_ && *it == value) {
            return 1;
        }
        return 0;
    }

    void clear() { pos = 0; }

    int size() const { return pos; }

    template <class T2, int N2>
    friend bool operator==(const SmallSet<T2, N2>& a,
                           const SmallSet<T2, N2>& b);
    template <class T2, int N2>
    friend bool operator!=(const SmallSet<T2, N2>& a,
                           const SmallSet<T2, N2>& b);

    typedef T* iterator;
    typedef const T* const_iterator;

    iterator begin() { return &data[0]; }
    const_iterator begin() const { return &data[0]; }
    iterator end() { return &data[pos]; }
    const_iterator end() const { return &data[pos]; }

private:
    std::array<T, N> data;
    int pos;
};

template <class T, int N>
bool operator==(const SmallSet<T, N>& a, const SmallSet<T, N>& b) {
    if (a.size() == b.size()) {
        return std::equal(a.data.begin(), a.data.begin() + a.pos,
                          b.data.begin());
    }
    return false;
}

template <class T, int N>
bool operator!=(const SmallSet<T, N>& a, const SmallSet<T, N>& b) {
    return !(a == b);
}

ASR_NAMESPACE_END