nvidia-smi
python ./code/train_ss_2d.py --labelnum 3 --gpu 0 && \
python ./code/test_ACDC.py --labelnum 3 && \
python ./code/train_ss_2d.py --labelnum 7 --gpu 0 && \
python ./code/test_ACDC.py --labelnum 7 
