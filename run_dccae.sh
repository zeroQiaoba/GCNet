set -e
dataset=$1

## train on cca
cd CCA-master
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize






python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize



python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=256 --max-epoch=100 --n-components=20 --normalize




## train on cca
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize






python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize



python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=512 --max-epoch=100 --n-components=20 --normalize



## train on cca
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.0 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.1 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize






python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.2 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize



python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.3 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.4 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.5 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize





python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.6 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize




python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=2
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=3
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=4
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=5
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=6
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=7
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=8
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=9
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=10
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=11
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=12
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=13
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=14
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=15
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=16
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=17
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=18
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=19
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=20
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=2 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=3 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=4 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=5 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=6 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=7 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=8 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=9 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=10 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=11 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=12 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=13 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=14 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=15 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=16 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=17 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=18 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=19 --normalize
python dccae.py --dataset=$dataset --missing-rate=0.7 --n-hidden=1024 --max-epoch=100 --n-components=20 --normalize

