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

#include <clocale>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

class PlyReader {
public:
    enum DataType {
        CHAR,
        UCHAR,
        SHORT,
        USHORT,
        INT,
        UINT,
        FLOAT,
        DOUBLE,
        UNKNOWN
    };
    enum Format { ASCII, BINARY_LE, BINARY_BE };
    struct Property {
	Property() : is_list(false), list_type(DataType::UNKNOWN), scalar_type(DataType::UNKNOWN), callback_idata(0) {};
        std::string name;
        bool is_list;
        DataType list_type;
        DataType scalar_type;
        std::function<void(std::vector<double>&, int)> callback;
        int callback_idata;
    };
    struct Element {
	Element() : count(0) {};
        std::string name;
        size_t count;
        std::vector<Property> properties;
        std::function<void(size_t, size_t)> callback;
    };
    struct Header {
	Header() : size(0),format(Format::ASCII) {};
        size_t size;
        Format format;
        std::vector<Element> elements;
    };

    PlyReader(const std::string& filename);
    ~PlyReader();

    Header readHeader();

    const Element* getElement(const std::string& element_name) const;
    const Property* getProperty(const std::string& element_name,
                                const std::string& property_name) const;

    bool setupCallback(std::function<void(size_t, size_t)> cb,
                       const std::string& element_name);
    bool setupCallback(std::function<void(std::vector<double>&, int)> cb,
                       const std::string& element_name,
                       const std::string& property_name,
                       int idata);

    size_t readElements(const std::string& element_name,
                        size_t limit = std::numeric_limits<size_t>::max());

    const std::string filename;

private:
    PlyReader(const PlyReader&) {}
    PlyReader& operator=(const PlyReader&) { return *this; }

    struct SetPlyLocale {
        SetPlyLocale() {
            prev_locale = setlocale(LC_NUMERIC, 0);
            setlocale(LC_NUMERIC, "C");
        }
        ~SetPlyLocale() { setlocale(LC_NUMERIC, prev_locale); }
        const char* prev_locale;
    };

    Element* getElement_(const std::string& element_name) const;
    Property* getProperty_(const std::string& element_name,
                           const std::string& property_name) const;

    DataType strToDataType(const std::string& str) {
        if (str == "char")
            return CHAR;
        else if (str == "uchar")
            return UCHAR;
        else if (str == "short")
            return SHORT;
        else if (str == "ushort")
            return USHORT;
        else if (str == "int")
            return INT;
        else if (str == "uint")
            return UINT;
        else if (str == "float")
            return FLOAT;
        else if (str == "double")
            return DOUBLE;
        else
            return UNKNOWN;
    }

    template <class T>
    inline T readValue(std::ifstream& file) {
        union {
            char data[sizeof(T)];
            T t;
        } tmp;
        switch (header.format) {
            case BINARY_LE:
                file.read(tmp.data, sizeof(T));
                break;
            case BINARY_BE:  // untested
            {
                file.read(tmp.data, sizeof(T));
                for (size_t i = 0; i < sizeof(T) / 2; ++i)
                    std::swap(tmp.data[i], tmp.data[sizeof(T) - i - 1]);
            } break;
            case ASCII: {
                // read as int if the type is (u)char
                if (std::is_same<T, char>::value ||
                    std::is_same<T, unsigned char>::value) {
                    int i;
                    file >> i;
                    tmp.t = i;
                } else
                    file >> tmp.t;
            } break;
        }
        return tmp.t;
    }

    template <class T>
    inline void skipValue(std::ifstream& file) {
        switch (header.format) {
            case BINARY_LE:
            case BINARY_BE:
                file.seekg(sizeof(T), file.cur);
                break;
            case ASCII: {
                // read as int if the type is (u)char
                if (std::is_same<T, char>::value ||
                    std::is_same<T, unsigned char>::value) {
                    int dummy;
                    file >> dummy;
                } else {
                    T dummy;
                    file >> dummy;
                }
            } break;
        }
    }

    inline void readProperty(std::vector<double>& values,
                             const Property& prop,
                             std::ifstream& file) {
        size_t value_count = 1;
        if (prop.is_list) {
            switch (prop.list_type) {
                case CHAR:
                    value_count = readValue<char>(file);
                    break;
                case UCHAR:
                    value_count = readValue<unsigned char>(file);
                    break;
                case SHORT:
                    value_count = readValue<short>(file);
                    break;
                case USHORT:
                    value_count = readValue<unsigned short>(file);
                    break;
                case INT:
                    value_count = readValue<int>(file);
                    break;
                default:
                case UINT:
                    value_count = readValue<unsigned int>(file);
                    break;
            }
        }
        values.resize(value_count);
        switch (prop.scalar_type) {
            case CHAR: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<char>(file);
            } break;
            case UCHAR: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<unsigned char>(file);
            } break;
            case SHORT: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<short>(file);
            } break;
            case USHORT: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<unsigned short>(file);
            } break;
            case INT: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<int>(file);
            } break;
            case UINT: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<unsigned int>(file);
            } break;
            case FLOAT: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<float>(file);
            } break;
            default:
            case DOUBLE: {
                for (size_t i = 0; i < value_count; ++i)
                    values[i] = readValue<double>(file);
            } break;
        }
    }

    inline void skipProperty(const Property& prop, std::ifstream& file) {
        size_t value_count = 1;
        if (prop.is_list) {
            switch (prop.list_type) {
                case CHAR:
                    value_count = readValue<char>(file);
                    break;
                case UCHAR:
                    value_count = readValue<unsigned char>(file);
                    break;
                case SHORT:
                    value_count = readValue<short>(file);
                    break;
                case USHORT:
                    value_count = readValue<unsigned short>(file);
                    break;
                case INT:
                    value_count = readValue<int>(file);
                    break;
                default:
                case UINT:
                    value_count = readValue<unsigned int>(file);
                    break;
            }
        }
        switch (prop.scalar_type) {
            case CHAR: {
                for (size_t i = 0; i < value_count; ++i) skipValue<char>(file);
            } break;
            case UCHAR: {
                for (size_t i = 0; i < value_count; ++i)
                    skipValue<unsigned char>(file);
            } break;
            case SHORT: {
                for (size_t i = 0; i < value_count; ++i) skipValue<short>(file);
            } break;
            case USHORT: {
                for (size_t i = 0; i < value_count; ++i)
                    skipValue<unsigned short>(file);
            } break;
            case INT: {
                for (size_t i = 0; i < value_count; ++i) skipValue<int>(file);
            } break;
            case UINT: {
                for (size_t i = 0; i < value_count; ++i)
                    skipValue<unsigned int>(file);
            } break;
            case FLOAT: {
                for (size_t i = 0; i < value_count; ++i) skipValue<float>(file);
            } break;
            default:
            case DOUBLE: {
                for (size_t i = 0; i < value_count; ++i)
                    skipValue<double>(file);
            } break;
        }
    }

    std::pair<std::ifstream*, size_t> createElementStream(
            const std::string& element_name) {
        std::pair<std::ifstream*, size_t> result;
        result.first = new std::ifstream(filename);
        result.second = 0;
        std::ifstream& file = *result.first;

        file.seekg(header.size);  // skip header

        for (Element element : header.elements) {
            if (element.name == element_name) return result;

            for (size_t element_idx = 0; element_idx < element.count;
                 ++element_idx) {
                for (const Property& prop : element.properties) {
                    skipProperty(prop, file);
                }
            }
        }

        file.close();
        throw std::runtime_error("createElementStream('" + element_name +
                                 "') failed");
        return std::pair<std::ifstream*, size_t>(0, 0);
    }

    mutable Header header;
    std::map<std::string, std::pair<std::ifstream*, size_t>> streams;
};

////////////////////////////////////////////////////////////////////////////////

inline PlyReader::PlyReader(const std::string& filename) : filename(filename) {}

inline PlyReader::~PlyReader() {
    for (auto it : streams) {
        it.second.first->close();
        delete (it.second.first);
    }
}

inline PlyReader::Header PlyReader::readHeader() {
    SetPlyLocale plylocale;

    std::ifstream file(filename.c_str());
    char plystr[4] = {0, 0, 0, 0};
    file.get(plystr, 4);
    if (plystr[0] != 'p' || plystr[1] != 'l' || plystr[2] != 'y')
        throw std::runtime_error("wrong file format");

    file.seekg(0);
    const int MAX_LINE_LEN = 256;
    char line[MAX_LINE_LEN];
    file.getline(line, MAX_LINE_LEN);

    Element current_element;
    while (true) {
        file.getline(line, MAX_LINE_LEN);
        std::string strline(line);

        std::istringstream isstrline;
        isstrline.str(strline);

        std::string word;
        isstrline >> word;
        if (word == "comment") {
            continue;
        } else if (word == "format") {
            isstrline >> word;
            if (word == "ascii")
                header.format = ASCII;
            else if (word == "binary_little_endian")
                header.format = BINARY_LE;
            else if (word == "binary_big_endian")
                header.format = BINARY_BE;
            else
                throw std::runtime_error("unknown file format: '" + word + "'");

            isstrline >> word;
            if (word != "1.0")
                throw std::runtime_error("unknown version: '" + word + "'");
        } else if (word == "element") {
            isstrline >> word;
            if (current_element.name.size())
                header.elements.push_back(current_element);

            current_element = Element();
            current_element.name = word;
            isstrline >> current_element.count;
        } else if (word == "property") {
            Property prop;
            isstrline >> word;
            if (word == "list") {
                prop.is_list = true;
                isstrline >> word;
                prop.list_type = strToDataType(word);
                if (prop.list_type == FLOAT || prop.list_type == DOUBLE)
                    throw std::runtime_error("list type '" + word +
                                             "' is not an integer type");
                if (prop.list_type == UNKNOWN)
                    throw std::runtime_error("unknown data type: '" + word +
                                             "'");
                isstrline >> word;
            } else {
                prop.is_list = false;
            }

            prop.scalar_type = strToDataType(word);
            isstrline >> word;
            prop.name = word;
            current_element.properties.push_back(prop);
        } else if (word == "end_header") {
            break;
        } else {
            throw std::runtime_error("unknown word: '" + word + "'");
        }
    }
    if (current_element.name.size()) header.elements.push_back(current_element);

    header.size = file.tellg();
    return header;
}

inline const PlyReader::Element* PlyReader::getElement(
        const std::string& element_name) const {
    return getElement_(element_name);
}

inline PlyReader::Element* PlyReader::getElement_(
        const std::string& element_name) const {
    for (Element& e : header.elements) {
        if (e.name == element_name) return &e;
    }
    return 0;
}

inline const PlyReader::Property* PlyReader::getProperty(
        const std::string& element_name,
        const std::string& property_name) const {
    return getProperty_(element_name, property_name);
}

inline PlyReader::Property* PlyReader::getProperty_(
        const std::string& element_name,
        const std::string& property_name) const {
    Element* element = getElement_(element_name);
    if (element) {
        for (Property& p : element->properties) {
            if (p.name == property_name) return &p;
        }
    }
    return 0;
}

inline bool PlyReader::setupCallback(std::function<void(size_t, size_t)> cb,
                                     const std::string& element_name) {
    Element* element = getElement_(element_name);

    if (element) {
        element->callback = cb;
        return true;
    }
    return false;
}

inline bool PlyReader::setupCallback(
        std::function<void(std::vector<double>&, int)> cb,
        const std::string& element_name,
        const std::string& property_name,
        int idata) {
    Property* property = getProperty_(element_name, property_name);
    if (property) {
        property->callback = cb;
        property->callback_idata = idata;
        return true;
    }
    return false;
}

inline size_t PlyReader::readElements(const std::string& element_name,
                                      size_t limit) {
    SetPlyLocale plylocale;

    // retrieve the element
    const Element* element = getElement(element_name);
    if (!element) return 0;

    // get the stream for this element
    if (!streams.count(element_name))
        streams[element_name] = createElementStream(element_name);

    std::pair<std::ifstream*, size_t> stream = streams[element_name];
    std::ifstream& file = *stream.first;
    size_t& elements_read = stream.second;

    std::vector<double> values;
    size_t i;
    for (i = 0; i < limit && elements_read < element->count;
         ++i, ++elements_read) {
        if (element->callback) element->callback(i, elements_read);
        for (const Property& prop : element->properties) {
            readProperty(values, prop, file);
            if (prop.callback) prop.callback(values, prop.callback_idata);
        }
    }
    streams[element_name] = stream;
    return i;
}
