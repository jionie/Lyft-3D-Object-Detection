set -x 
set -e

cd query_depth_point
rm -f *.so
python3 setup.py clean --all
cd ..

cd pybind11
rm -f *.so

