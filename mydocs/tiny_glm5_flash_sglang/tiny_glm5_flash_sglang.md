# Day 08｜Tiny GLM-5.3-Flash：从 PyTorch reference 到 SGLang CUDA runner

> 本文只记录当前仓库中的 tiny reference model 和 SGLang adapter。它不是 GLM-5.3-Flash 的精确 kernel 或 checkpoint 实现，也不加载原始 GLM 权重。
>
> 范围：随机权重、decoder-only、4 层、单 GPU smoke test。先解释 reference model 的精确数据流，再解释 SGLang adapter 怎样把同一层次结构接到已有 KDA、MLA/DSA、mHC 和 MoE runtime。SM120 与 SM86 的 backend 边界单独列出。

## 0. 先看最短主线

这份代码回答两个问题：

1. tiny 模型是否真的保留了 GLM-5.3-Flash 的主要结构？
2. 这些结构接入 SGLang 后，哪一部分是模型配置，哪一部分是硬件相关 backend？

最少只需要三个主角：

| 主角 | 真实对象 | 核心 shape / 内容 |
|---|---|---|
| hidden streams | `hidden_streams` | reference 中 `[B,T,4,H]`；SGLang DSV4 内部按 token 排列的 4 条 stream |
| attention branch | `TinyKDA` 或 `_TinyDSAAdapter` | 前三层 KDA recurrent state；第 4 层 MLA latent + DSA top-k |
| MLP branch | `TinySwiGLU` 或 `TinyMoE` / SGLang `DeepseekV2MoE` | 前三层 dense；第 4 层 routed experts + shared expert |

完整顺序如下：

~~~text
input_ids
→ embedding
→ 4 条 mHC residual streams
→ 每层 attention branch（KDA / KDA / KDA / DSA）
→ 每层 MLP branch（Dense / Dense / Dense / MoE）
→ final RMSNorm
→ stream collapse
→ lm_head
→ logits [B,T,4096]
~~~

reference model 的主入口是 [`TinyGlm5FlashForCausalLM`](../../scripts/tiny_glm5_flash.py#961-992)，SGLang 的 entry class 是 [`Glm5FlashTinyForCausalLM`](../../python/sglang/srt/models/glm5_flash_tiny.py#281-407)。

## 1. 配置：哪些字段负责缩小，哪些字段是 runtime 约束

配置定义在 [`TinyGlm5Config`](../../scripts/tiny_glm5_flash.py#48-154) 和 SGLang 侧的 [`Glm5FlashTinyConfig`](../../python/sglang/srt/configs/glm5_flash_tiny.py#30-221)。核心 tiny 字段是：

| 字段 | 当前值 | 作用 |
|---|---:|---|
| `vocab_size` | 4096 | embedding 行数和 `lm_head` 输出行数 |
| `hidden_size` | 256 | token hidden 宽度 |
| `num_hidden_layers` | 4 | 层数 |
| `num_attention_heads` | 4 | KDA/MLA query head 数 |
| `head_dim` | 64 | tiny 配置的通用 head 字段 |
| `intermediate_size` | 512 | dense SwiGLU 中间宽度 |
| `moe_intermediate_size` | 256 | 每个 routed/shared expert 的中间宽度 |
| `n_routed_experts` | 4 | 第 4 层 routed expert 数 |
| `n_shared_experts` | 1 | 第 4 层 always-active shared expert 数 |
| `num_experts_per_tok` | 2 | 每个 token 选中的 routed expert 数 |
| `index_topk` | 8 | DSA 选择的历史 token 数 |
| `hc_mult` | 4 | mHC residual stream 数，未缩成 2 |

层 schedule 是显式字段，不由 `num_hidden_layers` 猜测：

~~~python
layer_types = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]
mlp_layer_types = ["dense", "dense", "dense", "sparse"]
~~~

### 1.1 DSA 的 cache-facing 宽度不能随意改成 64

为了接入 SGLang 的 DSA FP8/KV pool，当前配置还保留了：

~~~text
kv_lora_rank       = 512
qk_nope_head_dim   = 512
qk_rope_head_dim   = 64
index_head_dim     = 128
~~~

这些字段不是把 DSA 换成普通 attention，而是 SGLang cache/kernel 的布局约束：

- FP8 DSA latent KV 的 quantization block 需要 `kv_lora_rank` 128 对齐；
- 当前 DSA FP8 cache 写入路径要求 latent non-rotary slice 为 512；
- indexer cache 的 `index_head_dim` 当前要求 128；
- rotary slice 必须非零，且当前 sparse cache 代码按 64 处理。

所以“缩小模型”主要缩小层数、hidden width、query heads、MLP 和 expert 数；不能把所有内部 cache width 都机械地改成 64，否则模型结构仍在，但 SGLang pool/kernel 在 forward 前就会拒绝 shape。

## 2. reference forward：四条 stream 怎样贯穿两条 branch

### 2.1 embedding 与 mHC 初始化

[`TinyGlm5FlashModel.forward()`](../../scripts/tiny_glm5_flash.py#905-958) 先把输入变成：

~~~text
input_ids      [B,T]
embed_tokens   [B,T,256]
expand stream  [B,T,4,256]
~~~

这里的 `expand` 是把同一个 embedding 视图复制到 4 条初始 stream；之后每一层的 mHC branch 会学习不同的 `pre`、`post` 和 `comb`。

### 2.2 mHC block 的两个站点

[`TinyMHCBlock.forward()`](../../scripts/tiny_glm5_flash.py#841-903) 每层执行两次 mHC：

~~~text
hidden_streams [B,T,4,256]
→ attn_hc.pre       collapse 成 [B,T,256]
→ attention branch
→ attn_hc.post/comb 写回 [B,T,4,256]
→ ffn_hc.pre        collapse 成 [B,T,256]
→ Dense 或 MoE
→ ffn_hc.post/comb 写回 [B,T,4,256]
~~~

[`TinyMHC.forward()`](../../scripts/tiny_glm5_flash.py#787-828) 中：

- `pre` 是每条 stream 的输入混合权重；
- `post` 是 branch 输出写回各 stream 的权重；
- `comb` 是 `4×4` stream combine matrix；
- `Sinkhorn` 迭代保持 combine 的近似流形约束。

因此 `hc_mult=4` 不是一个只影响参数量的整数；它同时决定输入 stream、输出 stream 和 `comb` 的最后两个维度。

### 2.3 KDA：recurrent state，而不是 token KV attention

前 3 层使用 [`TinyKDA`](../../scripts/tiny_glm5_flash.py#241-357)：

~~~text
x [B,T,256]
→ q/k/v projection
→ causal short convolution
→ per-head forget gate
→ recurrent state update
→ output projection
→ [B,T,256]
~~~

状态的核心 shape 是：

~~~text
recurrent_state [B, linear_num_heads=4, linear_head_dim=64, linear_head_dim=64]
~~~

每个时间步先衰减旧 state，再用当前 key/value 写入 delta，最后用 query 读出。`TinyKDA.last_recurrent_state` 会保存本次 forward 结束时的 detached state，便于 smoke test 检查 KDA 分支确实被执行。

SGLang 侧的 [`_TinyKDAAdapter`](../../python/sglang/srt/models/glm5_flash_tiny.py#33-82) 不重写 KDA 算法，而是把 DSV4 mHC layer 的调用约定转换成已有 [`KimiDeltaAttention`](../../python/sglang/srt/models/kimi_linear.py) 的调用约定。

### 2.4 DSA：indexer 先选位置，latent attention 再消费

第 4 层使用 [`TinyDSAIndexer`](../../scripts/tiny_glm5_flash.py#360-506) 和 [`TinyDSA`](../../scripts/tiny_glm5_flash.py#508-661)。它不是把所有历史 token 都做 dense attention，而是：

~~~text
hidden states
→ KPool 压缩历史 key
→ indexer score
→ 每个 query 选 index_topk=8 个历史位置
→ 对选中的 latent K/V 做 causal attention
~~~

`topk_indices` 的 reference shape 是：

~~~text
[B,T,index_topk + tail_width]
~~~

`tail_width` 用来保留尚未凑满一个 KPool 的 causal 尾部，所以短序列时最后一维可能大于 8。这个额外宽度不是把 `index_topk` 改大，而是把当前不完整 pool 的可见 token 保留下来。

DSA forward 会记录 `TinyDSA.last_topk_indices`；因此可以用它检查第 4 层确实执行了 sparse index，而不是误走 dense attention。

## 3. MoE：routed experts 和 shared expert 都保留

[`TinyMoE`](../../scripts/tiny_glm5_flash.py#756-785) 包含三段：

~~~text
router logits [B,T,4]
→ top-2 routed experts
→ 4 个 SwiGLU expert 中的稀疏聚合
→ 1 个 always-active shared expert
→ routed_output + shared_output
~~~

`TinyMoE.last_topk_indices` 和 `last_router_logits` 可用于检查 router 分支。SGLang 侧的 [`Glm5FlashTinyDecoderLayer`](../../python/sglang/srt/models/glm5_flash_tiny.py#165-214) 对前三层构造 `DeepseekV2MLP`，第 4 层沿用继承的 `DeepseekV2MoE`；[`shared_experts_fusion_disable_reason()`](../../python/sglang/srt/models/glm5_flash_tiny.py#288-293) 明确关闭 shared expert fusion，保留可读的 shared expert 模块。

## 4. SGLang adapter：配置不是唯一接入点

SGLang 的真实调用顺序是：

~~~text
config.json architectures=["Glm5FlashTinyForCausalLM"]
→ ModelRegistry 找到 EntryClass
→ Glm5FlashTinyForCausalLM
→ Glm5FlashTinyModel
→ Glm5FlashTinyDecoderLayer
→ KDA adapter / DSA adapter
→ HybridLinearAttnBackend + DSA backend
→ KDA/DSA kernel 或 tiny fallback
~~~

模型注册入口在 [`EntryClass`](../../python/sglang/srt/models/glm5_flash_tiny.py#407)，配置注册在 [`Glm5FlashTinyConfig`](../../python/sglang/srt/configs/glm5_flash_tiny.py#30-221)。因此从“模型结构”看，确实可以通过一个 config 选择已有组件；但从“runtime”看，仍需要：

- `model_config.py` 把 tiny architecture 识别为 MLA/DSA；
- hybrid backend 把 DSA indexer metadata 委托给 full-attention child；
- tiny top-k 走非 2048 的 index transform；
- tiny head 数适配 DeepGEMM 的 head-count 限制；
- 非生产形状选择 eager sparse-MLA fallback。

这些 glue code 不改变 KDA/DSA/MoE/mHC 的计算角色，只负责把已有模块接到正确的 runtime 生命周期。

### 4.1 为什么需要 `AttentionInputs`

[`_TinyDSAAdapter.forward()`](../../python/sglang/srt/models/glm5_flash_tiny.py#115-162) 显式安装：

~~~text
AttentionInputs(x, forward_batch, self.inner.prepare_qkv_latent)
~~~

原因是 DSV4 layer 直接调用 self-attention，而 DeepSeek-V2 MLA forward 原本依赖 `LayerCommunicator` 提前放入的 attention context。没有这一步，MLA/DSA 进入 `forward_prepare()` 时拿不到 latent q/kv 的准备函数。

### 4.2 为什么要关闭 fused mHC / fused MLA 小维度路径

tiny hidden 和 latent dimensions 会触发生产 fused kernel 的 shape 前提，例如：

- fused A GEMM 要求特定的 K 对齐；
- SM120 FlashInfer sparse MLA 只枚举生产 head/top-k/dimension 组合；
- index transform 的 Triton kernel默认按 `topk=2048` 编译。

因此 adapter 只在 tiny 路径关闭不适用的 fused optimization，保留 eager 计算；生产模型仍可走原有 fused path。

## 5. 缓存和地址：reference 与 SGLang 的边界

reference model 没有 SGLang request pool、KV cache、CUDA graph 或 distributed state。它在一次 `forward(input_ids)` 内直接计算：

~~~text
input_ids → embedding → all 4 layers → logits
~~~

SGLang 则会额外管理：

| runtime 对象 | 用途 |
|---|---|
| request row / token loc | Full/DSA token cache 的地址映射 |
| Mamba/KDA state pool | KDA recurrent state 的 slot 生命周期 |
| DSA main KV pool | MLA latent + rotary cache |
| DSA indexer sidecar | index key + scale，产生 top-k 候选 |
| `ForwardBatch` | 当前 extend/decode 的临时 metadata |

在当前 tiny SGLang smoke 脚本中，`page_size=64` 是 DSA pool 的要求，`mamba_radix_cache_strategy="extra_buffer"` 用来让 hybrid mamba cache 与 page-size=64 共存。

## 6. GPU backend：SM120 与 SM86 不能共用一套命令

### 6.1 已验证的 SM120 路径

RTX 5060 Ti 的 SM120 smoke 使用：

~~~text
KDA: Triton
DSA KV cache: FP8 E4M3
DSA prefill/decode: flashinfer_sparse_mla
tiny unsupported sparse shape: eager Torch fallback
MoE: SGLang default Triton fallback
mHC: eager path
~~~

`flashinfer_sparse_mla_forward()` 在 [`flash_mla_sm120.py`](../../python/sglang/kernels/ops/attention/flash_mla_sm120.py#610-766) 中先尝试 FlashInfer；如果 kernel dispatch table 没有 tiny shape，就进入 `_torch_sparse_mla_forward()`，直接解码 DSA packed cache 并做 selected-token attention。

### 6.2 SM86 当前需要另一条路径

SM86 没有当前 SM120 的 GLM FP8 sparse-MLA 路径，因此不能直接使用 smoke 脚本里的：

~~~python
kv_cache_dtype="fp8_e4m3"
dsa_prefill_backend="flashinfer_sparse_mla"
dsa_decode_backend="flashinfer_sparse_mla"
~~~

SM86 需要单独处理：

1. KV cache 改为 BF16，避免依赖 SM120 FP8 sparse layout；
2. DSA indexer 不依赖 DeepGEMM paged-MQA logits，改成 Torch/Triton eager index score；
3. sparse attention 直接 gather paged KV，再对 `index_topk=8` 做 Torch/Triton attention；
4. KDA 继续使用 Triton；
5. mHC 和 MoE 继续走 eager/default fallback。

当前仓库已经有 SM120 tiny fallback，但它读取的是 SM120 FP8 packed cache，不能冒充 SM86 backend。换句话说：

~~~text
reference CUDA forward       ✅ SM86 可直接尝试
SGLang SM120 smoke script    ✅ SM120
SGLang SM86 DSA runner       ⏳ 需要 BF16 + indexer/attention fallback
~~~

## 7. 运行与验收

### 7.1 reference model

语法检查：

~~~bash
python -m py_compile scripts/tiny_glm5_flash.py
~~~

CPU：

~~~bash
python scripts/tiny_glm5_flash.py \
  --device cpu \
  --seq-len 8 \
  --batch-size 2 \
  --save-dir /tmp/tiny-glm5-flash
~~~

期望：

~~~text
input_ids shape: [2, 8]
logits shape: [2, 8, 4096]
~~~

CUDA：

~~~bash
python scripts/tiny_glm5_flash.py \
  --device cuda \
  --dtype bfloat16 \
  --seq-len 16
~~~

### 7.2 SGLang SM120 smoke

先让 reference 脚本写出 `config.json`，再让 SGLang 用 dummy loader 随机初始化自己的 runtime 权重：

~~~bash
python scripts/tiny_glm5_flash.py \
  --device cuda \
  --dtype bfloat16 \
  --seq-len 8 \
  --save-dir /tmp/tiny-glm5-flash-sglang

PYTHONPATH=python python scripts/tiny_glm5_flash_sglang.py \
  --model-dir /tmp/tiny-glm5-flash-sglang \
  --batch-size 2 \
  --seq-len 8 \
  --dtype bfloat16
~~~

`pytorch_model.bin` 供 reference state-dict reload 验证；SGLang smoke 使用 `load_format="dummy"`，不会误把 reference state dict 当成生产 GLM checkpoint。

## 8. 当前不变量与边界

已经验证的 reference 不变量：

- `input_ids` 全部小于 `vocab_size`；
- `embed_tokens.weight.shape == [4096,256]`；
- `lm_head.weight.shape == [4096,256]`；
- batch size 和 sequence length 都可大于 1；
- `TinyKDA.last_recurrent_state` 非空；
- 第 4 层 `TinyDSA.last_topk_indices` 非空；
- 第 4 层 `TinyMoE` 有 routed experts 和 shared expert；
- 每个 `TinyMHCBlock.mhc_forward_count` 都增加；
- 保存后的 reference state dict 可严格重新加载。

当前限制：

- 随机权重只用于结构调试，不代表模型效果；
- reference `pytorch_model.bin` 与 SGLang dummy model 的参数命名/权重加载不是同一条生产 checkpoint 转换链；
- SM120 eager sparse fallback 是可读参考实现，不是 GLM-5.3-Flash 的真实 kernel；
- SM86 的 SGLang DSA runner 还需要 BF16 + Torch/Triton backend 适配；
- 当前没有把 tiny model 注册成正式的 GLM-5 模型，也没有修改原始 GLM registry。

后续如果要支持 SM86，最小改动应集中在 DSA backend 的三层：`indexer score`、`paged KV gather`、`selected-token attention`。KDA、mHC、MoE 和四层 schedule 不需要再复制一套模型结构。
