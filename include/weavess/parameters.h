//
// Created by Murph on 2020/8/12.
//

#ifndef WEAVESS_PARAMETERS_H
#define WEAVESS_PARAMETERS_H

#include <unordered_map>
#include <sstream>
#include <typeinfo>

// inline const ?

namespace weavess{
    class Parameters{
    public:
        template<typename T>
        inline void Set(const std::string &name, const T &value){
            std::stringstream sstream;
            sstream << value;
            params[name] = sstream.str();
        }

        template<typename T>
        inline T Get(const std::string &name) const{
            auto item = params.find(name);
            if(item == params.end()){
                throw std::invalid_argument("Invalid paramter name.");
            }else{
                return ConvertStrToValue<T>(item->second);
            }
        }

        template<typename T>
        inline T Get(const std::string &name, const T &default_value){
            try {
                return Get<T>(name);
            } catch (std::invalid_argument e) {
                return default_value;
            }
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template<typename T>
        inline T ConvertStrToValue(const std::string &str) const{
            std::stringstream sstream(str);
            T value;
            if(!(sstream >> value) || !sstream.eof()){
                std::stringstream err;
                err << "Fail to convert value" << str << " to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }

            return value;
        }
    };
}

#endif //WEAVESS_PARAMETERS_H
