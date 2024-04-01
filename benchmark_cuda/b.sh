cd ./fff_cuda_folderOG
python setup.py clean install
cd ..
cd ./fff_cuda_folder
python setup.py clean install
cd ..
python test.py fff-cuda --batch-size 16384 --input-width 1024 --depth 12 --runs 10 --scale us --cuda
