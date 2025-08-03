# Notes
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])

##
How did 1024 get fixed in training - and how modern architecture change

## Residual Layer std normalisation
Normalisation  x**-0.5 because residual layer tends to expand std (always 1+ something)


##
Weight Sharing between linear classifier head and token embedding (`wpe`)

```python
# weight sharing scheme
self.transformer.wte.weight = self.lm_head.weight
```

## speed up
- `torch.set_float_32_matmul_precision("high")` # Default is "highest"
- `torch.compile`
- vocab size from 50257 -> 50304 -> 4% speedup because of Kernel divergence

## Parallel computing
- Use `torch.nn.DistributedDataParallel` - bye `torch.nn.DataParallel`
- Gradient Accumulationcode https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

How context window change from 1024 to 1 million in modern architecture?
- Why is it 1024? How does the training process affect this length?

Why GPT does not use `bias` for the linear layer?

Redundancy due to normalization: In transformer architectures, each sub-layer (including those with linear projections) is typically followed or preceded by a normalization layer, such as LayerNorm or RMSNorm. These normalization steps re-center and re-scale the outputs, effectively removing the effect of any bias term in the preceding linear layer. As a result, including a bias would be computationally redundant, adding parameters and memory footprint without any functional gain.

Why 50257 Tokens?

50000 BPE merge + 256 bytes + 1 endoftext token
(But what is BPE merge and 256 bytes token?)

