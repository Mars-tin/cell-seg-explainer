python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy
pip install pickle
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==latest+cpu torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch_geometric
cp bin/__init__.pyi env/lib/python*/site-packages/torch/optim/__init__.pyi
