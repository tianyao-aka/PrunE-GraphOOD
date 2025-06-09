To run our method, here is an example for GOODMotif dataset:

```
python run.py --device 0 --batch_size 64 --dataset goodmotif --domain basis --shift covariate --base_gnn gin --early_stop_epochs 50  --nhid 128 --epochs 100 --pretraining_epochs 10 --nlayers 4 --edge_gnn_layers 2  --edge_gnn gin --edge_uniform_penalty 0.001 --edge_prob_thres 90 --edge_budget 0.85 --edge_penalty 10   --seed 1 --with_bn --dropout 0.0
```

For GOODMotif datasets, the domain can be chosen among: _{basis, size}_; For GOODHIV datasets, the domain can be chosen among: {scaffold,size}. For OGBG datasets, the domain can also be: {scaffold,size}.


`edge_gnn` corresponds to the subgraph selector $t(\cdot)$, `edge_gnn_layers` denotes the #layers in $t(\cdot)$. `edge_uniform_penalty`  corresponds to $\lambda_2$ in the paper, `edge_prob_thres`  corresponds to $K$ in our paper, `edge_budget` corresponds to $\eta$ in our paper, `edge_penalty`  corresponds to $\lambda_1$ in our paper. For GOODHIV and OGBG dataset, `--mol_encoder` should be added to the command line to use AtomEncoder and BondEncoder.

The software versions are listed below:

```
torch==2.1.2
torch_geometric==2.4.0
numpy==1.24.4
ogb==1.3.6
```

To install the dependencies for GOOD, follow the instruction [here](https://good.readthedocs.io/en/latest/installation.html).

