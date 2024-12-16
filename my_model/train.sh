# Initialize conda
python3 /root/autodl-tmp/project/train.py -m my_mbt2018 -d /root/autodl-tmp/dataset --epochs 500 -lr 1e-4 --lambda 0.0075 --batch-size 16 --cuda --save > log3.out && /usr/bin/shutdown
