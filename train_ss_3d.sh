nvidia-smi
python ./code/train_ss_3d.py --labelnum 4 --gpu 0 && \
python ./code/test_LA.py --labelnum 4  && \
python ./code/train_ss_3d.py --labelnum 8 --gpu 0 && \
python ./code/test_LA.py --labelnum 8
