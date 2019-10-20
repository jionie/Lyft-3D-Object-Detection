#!/bin/bash
set -x 
set -e 

include=`python3 -m pybind11 --includes`

g++ -std=c++11 -shared -o box_ops_cc.so box_ops.cc -fPIC -O3 ${include}

