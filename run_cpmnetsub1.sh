set -e
dataset=$1

cd cpm-net
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=300 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=20 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=30 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=20 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=20 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=30 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=300 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=50 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=500 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=50 --epochs-test=80 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=100 --epochs-test=30 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=300 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=200 --epochs-test=30 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=260 --lsd-dim=512 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=128 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=150 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=256 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=512 --lamb=1
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=128 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=150 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=256 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=512 --lamb=10
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=128 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=150 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=256 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=512 --lamb=1 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=128 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=150 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=256 --lamb=10 --normalize
python -u test_lianzheng.py --dataset=$dataset --missing-rate=0.0 --epochs-train=90 --epochs-test=100 --lsd-dim=512 --lamb=10 --normalize



