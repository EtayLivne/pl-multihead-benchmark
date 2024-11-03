# pl-multihead-benchmark
Benchmark to showcase performance degradation in pytorch lightning where models have multiple losses/metrics

## How to use

In an enviroment with Lightning 2.4.0 installed and access to a GPU, run 

python benchmark_pure_torch.py

and 

python benchmark_with_lightning.py

to compare how changing the number of metrics/losses from 1 to 2 to 4 impacts steps per second performance in each case.
While there's a high degree of variance, the lightning case faily consistently shows serious degradation as the number of metrics/losses increases, and vanilla torch appears not to. 


## Benchmark specifications

The benchmark takes a model with about 72 million weights across several linear layers, where the last layer has 4 weight. I compare the case where these weights are considered a single output, with one metric and one loss, vs cases where I have two losses + metrics, and four losses + metrics. I've found what appears to be an increasing performance degradation with additional metrics and losses, accumulating to as much as 40-50 % in some runs. The same does not appear to replicate in equivalent training loops I've set up with vanilla Pytorch. 

The lightning benchmark is set up to run with barebones=True, on a machine with 2 CPUs and one GPU (container on an A100). The data is a set of in-memory random vectors with random labels. I've written the models and metrics as "flat" as I can, not using any nested structures to hold them (no lists/dicts, no nn.Sequential, etc.). The training loop is of 10_000 steps, in a single training epoch and not including a validation epoch. I use seed_everything to make the experiment deterministic across runs, in both the vanilla torch and lightning case. 
