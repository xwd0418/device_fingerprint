source /opt/conda/bin/activate base
nvidia-smi -L
nvcc --version
cat /usr/local/cuda/version.txt
nvidia-smi
python3 -m pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python3 -c "import torch; print(torch.__version__)"
python3 -m pip install torch==2.0.4
python3 -m pip install torch-lightning==2.0.4
python3 -m pip install pytorch-lightning==2.0.4
pip list | grep torch
pip list | casa
pip list | grep casa
pip uninstall casanovods
python3 -m pip install pytorch-lightning==2.0.4
pip install pytorch-lightningsource /opt/conda/bin/activate base
pip install pytorch-lightning
pip list | grep light
pip list | grep torch
