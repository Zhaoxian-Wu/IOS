# Byzantine-Resilient Decentralized Stochastic Optimization
This hub stores the code for paper *Byzantine-Resilient Decentralized Stochastic Optimization with Robust Aggregation Rules* [[Arxiv Link](https://arxiv.org/abs/2206.04568)].

## Install
1. Download the dependant packages (c.f. `install.sh`):
- python 3.8.10
- pytorch 1.9.0
- matplotlib 3.3.4
- networkx 2.5.1

2. Download the dataset to the directory `./dataset` and create a directory named `./record`. The experiment outputs will be stored in `./record`.

- *MNIST*: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Construction
The main programs can be found in the following files:
- `ByrdLab`: main codes
- `main DSGD.py`, `main RSA.py`: program entry
- `run-twocastle.py`, `run-ER.py`, `run-octopus.py`: batch processing files
- `draw_table`, `draw_decentralized_one_fig`, `draw_decentralized_multi_fig`: directories containing the codes that draw the figures in paper
`record`: directories containing the experiment results
- `Demo`: some demos for potential developers



## Runing
### Run DSGD
```bash
python "main DSGD.py" --graph <graph-name> --aggregation <aggregation-name> --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main DSGD.py" --graph TwoCastle --aggregation trimmed-mean --attack sample_duplicate --data-partition noniid
```

> The arguments can be
>
> `<graph-name>`: 
> - CompleteGraph
> - TwoCastle
> - ER
> - OctopusGraph
>
> `<aggregation-name>`: 
> - no-comm
> - mean
> - trimmed-mean
> - median
> - geometric-median
> - faba
> - Krum
> - mKrum
> - bulyan
> - cc
> - scc
>
> `<attack-name>`: 
> - sign_flipping
> - gaussian
> - isolation
> - sample_duplicate
>
> `<data-partition>`: 
> - trival
> - iid
> - noniid

---
### Run RSA
```bash
python "main RSA.py" --graph <graph-name> --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main RSA.py" --graph TwoCastle --attack sample_duplicate --data-partition noniid
```

---


## Batch Running
```bash
# Table II-III
python run-twocastle.py
# Table IV
python run-octopus.py
# Table V-VI
python run-ER.py
```

## Result
```bash
# Table II-III, V-VI
cd draw_table
python draw.py --task SR_mnist --graph TwoCastle_k=6_b=2_seed=40 --partition iidPartition
python draw.py --task SR_mnist --graph TwoCastle_k=6_b=2_seed=40 --partition LabelSeperation
python draw.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition iidPartition
python draw.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition LabelSeperation
# Table IV
python draw_octopus.py --task SR_mnist --graph Octopus_head=6_headb=0_handb=2 --partition LabelSeperation

# ====================
# Fig. 2
cd draw_decentralized_one_fig
python draw.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition LabelSeperation --portrait
# python draw-LF.py --task NeuralNetwork_cifar10 --graph Complete_n=10_b=2 --partition LabelSeperation --portrait
```