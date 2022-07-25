conda create -n dev python=3.9
conda activate dev
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install numpy pandas scikit-learn transformers datasets tensorflow streamlit flask flask_restful jupyterlab