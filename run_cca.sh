set -e
dataset=$1

## train on cca
cd CCA-master
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.0 --n-components=20 --normalize





python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.1 --n-components=20 --normalize






python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.2 --n-components=20 --normalize



python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.3 --n-components=20 --normalize




python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.4 --n-components=20 --normalize




python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.5 --n-components=20 --normalize





python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.6 --n-components=20 --normalize




python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=2
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=3
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=4
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=5
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=6
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=7
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=8
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=9
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=10
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=11
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=12
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=13
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=14
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=15
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=16
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=17
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=18
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=19
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=20
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=2 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=3 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=4 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=5 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=6 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=7 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=8 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=9 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=10 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=11 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=12 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=13 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=14 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=15 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=16 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=17 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=18 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=19 --normalize
python cca.py --dataset=$dataset --missing-rate=0.7 --n-components=20 --normalize