# 4.1｜Tiny GLM-5.3-Flash：从 PyTorch reference 到 SGLang CUDA runner

> 本文只记录当前仓库中的 tiny reference model 和 SGLang adapter。它不是 GLM-5.3-Flash 的精确 kernel 或 checkpoint 实现，也不加载原始 GLM 权重。
>
> 范围：随机权重、decoder-only、4 层、单 GPU smoke test。先解释 reference model 的精确数据流，再解释 SGLang adapter 怎样把同一层次结构接到已有 KDA、MLA/DSA、mHC 和 MoE runtime。SM120 与 SM86 的 backend 边界单独列出。

如果只想先认识 SGLang 的整体请求主线，请先读 [0｜SGLang Serving State 对象地图的总览](../0_sglang_state_object_atlas/0_sglang_state_object_atlas.md)，再回到本文的 GLM5 专项路径。

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

reference model 的主入口是 [`TinyGlm5FlashForCausalLM`](../../scripts/tiny_glm5_flash.py#965-996)，SGLang 的 entry class 是 [`Glm5FlashTinyForCausalLM`](../../python/sglang/srt/models/glm5_flash_tiny.py#281-407)。

## 1. 配置：哪些字段负责缩小，哪些字段是 runtime 约束

配置定义在 [`TinyGlm5Config`](../../scripts/tiny_glm5_flash.py#48-162) 和 SGLang 侧的 [`Glm5FlashTinyConfig`](../../python/sglang/srt/configs/glm5_flash_tiny.py#30-221)。核心 tiny 字段是：

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
| `index_n_heads` | 8 | tiny DSA indexer 的最小 DeepGEMM 支持 head 数 |
| `index_kpool` | 4 | 连续 4 个 token 组成一个 index group |
| `index_kpool_compress` | `True` | 用学习到的 gate 把组内 4 个 index key 压成一个 pooled key |
| `index_kpool_always_select_tail` | `True` | 强制保留尚未凑满 4 个 token 的 causal tail |
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

为了接入 SGLang 的 DSA/KV pool（SM120 走 FP8，SM80/SM86 走 BF16），当前配置还保留了：

~~~text
kv_lora_rank       = 512
qk_nope_head_dim   = 512
qk_rope_head_dim   = 0
index_head_dim     = 128
~~~

这些字段不是把 DSA 换成普通 attention，而是 SGLang cache/kernel 的布局约束：

- FP8 DSA latent KV 的 quantization block 需要 `kv_lora_rank` 128 对齐；Ampere 的 BF16 路径也沿用相同 latent 宽度；
- 当前 DSA latent cache 写入路径要求 latent non-rotary slice 为 512；SM120 量化为 FP8，SM80/SM86 保持 BF16；
- indexer cache 的 `index_head_dim` 当前要求 128；
- GLM-5.3-Flash 的 DSA 是 no-RoPE 路径，`qk_rope_head_dim` 必须保持为 0；
  cache 中不会人为追加 64-wide rotary tail。

所以“缩小模型”主要缩小层数、hidden width、query heads、MLP 和 expert 数；不能把所有内部 cache width 都机械地改成 64，否则模型结构仍在，但 SGLang pool/kernel 在 forward 前就会拒绝 shape。

## 2. reference forward：四条 stream 怎样贯穿两条 branch

### 2.1 embedding 与 mHC 初始化

[`TinyGlm5FlashModel.forward()`](../../scripts/tiny_glm5_flash.py#909-962) 先把输入变成：

~~~text
input_ids      [B,T]
embed_tokens   [B,T,256]
expand stream  [B,T,4,256]
~~~

这里的 `expand` 是把同一个 embedding 视图复制到 4 条初始 stream；之后每一层的 mHC branch 会学习不同的 `pre`、`post` 和 `comb`。

### 2.2 mHC block 的两个站点

[`TinyMHCBlock.forward()`](../../scripts/tiny_glm5_flash.py#845-906) 每层执行两次 mHC：

~~~text
hidden_streams [B,T,4,256]
→ attn_hc.pre       collapse 成 [B,T,256]
→ attention branch
→ attn_hc.post/comb 写回 [B,T,4,256]
→ ffn_hc.pre        collapse 成 [B,T,256]
→ Dense 或 MoE
→ ffn_hc.post/comb 写回 [B,T,4,256]
~~~

[`TinyMHC.forward()`](../../scripts/tiny_glm5_flash.py#791-831) 中：

- `pre` 是每条 stream 的输入混合权重；
- `post` 是 branch 输出写回各 stream 的权重；
- `comb` 是 `4×4` stream combine matrix；
- `Sinkhorn` 迭代保持 combine 的近似流形约束。

因此 `hc_mult=4` 不是一个只影响参数量的整数；它同时决定输入 stream、输出 stream 和 `comb` 的最后两个维度。

### 2.3 KDA：recurrent state，而不是 token KV attention

前 3 层使用 [`TinyKDA`](../../scripts/tiny_glm5_flash.py#245-361)：

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

第 4 层使用 [`TinyDSAIndexer`](../../scripts/tiny_glm5_flash.py#364-512) 和 [`TinyDSA`](../../scripts/tiny_glm5_flash.py#515-644)。它不是把所有历史 token 都做 dense attention，而是：

~~~text
hidden states
→ 产生逐 token index key
→ 每 4 个 key 学习式压缩成一个 pooled key
→ query 给所有完整 pool 打分
→ 选 index_topk / index_kpool = 2 个 pool
→ 展开回 8 个原始 token 位置，再追加未完整 tail
→ 对选中位置的原始 latent K/V 做 causal attention
~~~

这里的 KPool 是 **indexer 的逻辑分组**，不是 `TokenToKVPool` 那个内存池。pooled key 只用来找候选位置；真正的 sparse MLA 仍然读取选中 token 的 main latent KV，不会拿 pooled key 当 attention key/value。

#### 2.4.1 逐 token key 怎样变成 pooled key

[`TinyDSAIndexer._pool_keys()`](../../scripts/tiny_glm5_flash.py#411-447) 先生成逐 token index key 和压缩 gate：

~~~python
keys = self.k_norm(self.wk(hidden_states))
gates = self.index_kpool_compress_gate(hidden_states)
grouped_keys = keys.view(batch_size, num_pools, index_kpool, index_head_dim)
pool_logits = grouped_gates + self.index_kpool_compress_ape
probabilities = softmax(pool_logits, dim=2)
pool_keys = (probabilities * grouped_keys).sum(dim=2)
~~~

上面是对真实源码的省略分支摘录。设第 $p$ 个 pool 内的第 $r$ 个 token、index 维度 $d$ 上的 key 为 $k_{p,r,d}$，则压缩权重和 pooled key 是：

$$
\alpha_{p,r,d}
=
\operatorname{softmax}_{r}
\left(g_{p,r,d}+a_{r,d}\right),
\qquad
\bar{k}_{p,d}
=
\sum_{r=0}^{3}\alpha_{p,r,d}k_{p,r,d}.
$$

`index_kpool_compress_gate` 产生 $g$，`index_kpool_compress_ape` 是组内位置参数 $a$。因此这不是四个 key 的简单平均，而是按维度学习的加权压缩。现在手里的 `pool_keys` 是 `[B,num_pools,128]`，下一步才由 query 给它们打分。

#### 2.4.2 改的是候选 key 和 top-k 粒度，不是改成随机选择

[`TinyDSAIndexer.forward()`](../../scripts/tiny_glm5_flash.py#449-512) 的分数主体仍然是 DSA 的多 index-query-head 匹配：

$$
s_{t,p}
=
\sum_h w_{t,h}
\operatorname{ReLU}
\left(
\frac{q_{t,h}\cdot\bar{k}_p}{\sqrt{D_I}}
\right).
$$

普通逐 token DSA 在上式中使用 $k_i$、对 token $i$ 排序；GLM KPool 改成使用 $\bar{k}_p$、对 pool $p$ 排序。`torch.topk` 始终选分数最高的 pool，没有随机采样。当前 tiny 使用随机初始化权重，所以选中哪个 pool 没有语义，但固定 seed 后结果仍是确定的。

关键变量直接对应源码：

| 变量 | tiny 值 | 代码含义 |
|---|---:|---|
| `index_kpool` | 4 | 一个完整 pool 包含 4 个原始 token |
| `index_topk` | 8 | 选中 pool 展开后的 token 预算，不包括 tail |
| `pools_to_select` | $8/4=2$ | 实际对 pool score 做的 top-k 数 |
| `complete_count` | $\lfloor L/4\rfloor$ | 长度 $L$ 下可参与打分的完整 pool 数 |
| `tail_count` | $L\bmod4$ | 未完整 pool 中强制保留的 0～3 个 token |
| `output_width` | $8+(4-1)=11$ | `topk_indices` 固定分配宽度，空位用 `-1` 补齐 |

例如 $L=14$ 时，有三个完整 pool 和两个 tail token：

~~~text
P0=[0,1,2,3]  P1=[4,5,6,7]  P2=[8,9,10,11]  tail=[12,13]
~~~

Indexer 在 `P0/P1/P2` 中按分数选 2 个，展开为 8 个 token，再强制追加 `12,13`。也就是最终参加 sparse MLA 的是 10 个原始 token，而不是 2 个 pooled key。

#### 2.4.3 多长才真正淘汰 token

设 $K=\text{index\_topk}$、$P=\text{index\_kpool}$，最后一个未完整 pool 最多有 $P-1$ 个 token，因此全选仍然能容纳的最大上下文长度是：

$$
L_{\text{all-visible,max}}=K+P-1.
$$

tiny 中这个值是 $8+4-1=11$：

| 可见长度 | 结果 |
|---:|---|
| $L\le 11$ | 候选预算足以放下全部可见 token，没有实际裁剪 |
| $L\ge 12$ | 完整 pool 数超过 2，开始按分数淘汰整个 pool |

GLM-5.3-Flash 生产配置是 `index_topk=2048`、`index_kpool=4`，同理在 $L\ge2052$ 时才开始真正裁剪。但 DSA 层和 Indexer 的存在与 $L$ 无关：本 tiny 的第 4 层始终是 DSA，只是短序列时 top-k 等价于全选。

`topk_indices` 的 reference shape 因此是：

~~~text
[B,T,index_topk + index_kpool - 1] = [B,T,11]
~~~

DSA forward 会记录 `TinyDSA.last_topk_indices`；因此可以用它检查第 4 层确实执行了 sparse index，而不是误走 dense attention。

#### 2.4.4 PyTorch reference 与 SGLang KPool 逐项对照

这两份代码对齐的是 GLM-5.3-Flash 的 DSA 数学结构，不是要求 eager PyTorch 逐行复刻 serving kernel。构造阶段可直接对照 [`TinyDSAIndexer.__init__`](../../scripts/tiny_glm5_flash.py#373-403) 和 [`IndexerKPool.__init__`](../../python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py#56-165)：

| 环节 | PyTorch reference | SGLang KPool | 对齐结论 |
|---|---|---|---|
| index key | `wk(hidden_states)` | `wk(x)` | 同一个无 bias 线性投影 |
| index key norm | `nn.LayerNorm(128)` | `LayerNorm(128, dtype=float32)` | 都是带 weight、bias 的 LayerNorm；不是 decoder block 使用的 RMSNorm |
| KPool 压缩 | `_pool_keys()` 中按组 softmax 后加权求和 | `kpool_*softmax_rotate_write_cache` kernel | gate、APE、组内 softmax、加权和的数学含义相同 |
| index query 与 head gate | `q_b_proj`、`weights_proj` | `wq_b`、`weights_proj` | 投影和多 head 聚合公式相同 |
| top-k | `torch.topk` | 固定 bucket kernel；tiny 的 `group_topk=2` 回退到 Torch backend | 都按 score 确定性选择，不是随机抽取 |
| no-RoPE | `qk_rope_head_dim=0`，不构造 rotary slice | GLM 层以 `skip_rope=True` 构造 Indexer | 语义一致 |
| 历史状态 | 每次 forward 从当前完整序列重新 `_pool_keys()` | 完整 pool 写 `IndexKeyCache`，不完整 group 留在 request tail | 数学结果对应，生命周期实现不同 |
| serving 数值路径 | 保持普通 PyTorch tensor | SM120 pooled key 做 Hadamard+FP8；SM80/SM86 pooled key 做 Hadamard+BF16 | rotation 同时作用于 query/key；Ampere 不依赖 FP8 |

`k_norm` 很容易因 GLM 其他位置大量使用 RMSNorm 而抄错。当前 reference 已改为带 bias 的 `nn.LayerNorm`，对应 SGLang 的 [`self.k_norm = LayerNorm(...)`](../../python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py#152)。SGLang 还显式让该层以 FP32 参数/累积运行，这是 serving 数值稳定性细节，不改变归一化定义。

因此“除了规模外对齐”应理解为：层次结构、投影、归一化定义、KPool 压缩公式、候选分数和 top-k 语义一致；reference 不模拟分页持久缓存、Triton kernel、Hadamard 后的具体 FP8/BF16 cache layout 与 CUDA graph。后几项属于运行时实现边界，不应硬塞进教学用 eager forward。

## 3. MoE：routed experts 和 shared expert 都保留

[`TinyMoE`](../../scripts/tiny_glm5_flash.py#760-788) 包含三段：

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
- all-visible smoke 的最终 index transform 支持非 2048 宽度；score-based `group_topk=2` 在未匹配生产 bucket 时改走 Torch eager top-k；
- tiny head 数适配 DeepGEMM 的 head-count 限制；
- 非生产形状的 formal sparse-MLA 选择 eager fallback，pooled indexer 的小 `group_topk` 也有独立 Torch fallback；
- 生产 `topk=2048`、`group_topk=512` 仍走已编译路径。

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
| DSA main KV pool | MLA latent cache（GLM-5.3-Flash 无 rotary tail） |
| DSA `IndexKeyCache` sidecar | 完整 KPool 的 pooled index key + scale |
| DSA compress tail | 每请求、每 DSA 层暂存未凑满 4 个 token 的 key/gate |
| `ForwardBatch` | 当前 extend/decode 的临时 metadata |

文档基础命令和 `.vscode/launch.json` 都显式传入 `seq_len=8`，因此该 smoke 只占一个 64-token page。`seq_len=68` 的两页、真实 pool 裁剪路径也已单独验证。`mamba_radix_cache_strategy="extra_buffer"` 用来让 hybrid mamba cache 与 page-size=64 共存。

### 5.1 page、KPool 和两种“pool”不是同一层

`DSATokenToKVPool` 是持有 main KV 和 index sidecar 的内存 owner；`index_kpool=4` 则是 Indexer 算法的分组宽度。SGLang 不会因为 KPool 再建立第二套 page allocator，pooled page table 从 main 的 `real_page_table` 派生。

| 层次 | 生产者 | owner / carrier | 消费者 |
|---|---|---|---|
| main latent KV | MLA Q/KV prepare + DSA backend | `DSATokenToKVPool` 的 main buffer | sparse MLA |
| pooled index key | `IndexerKPool._compress_write*` | `DSATokenToKVPool.index_key_cache` | paged/ragged MQA logits kernel |
| incomplete tail | 新 token 的 index key + compress gate | `_compress_tail_k/_compress_tail_score` | 下一次 decode 凑满 pool 时的 compress kernel |
| pooled 寻址信息 | `init_pooled_paged_mqa_metadata` | `DSAMetadata.pooled_*` | `IndexerKPool._get_topk_paged/ragged` |
| 最终 token 位置 | pooled score top-k + group expand + tail append | `topk_indices` | sparse MLA backend |

SGLang 的增量生命周期可以先用这段教学等价伪代码闭环理解：

~~~text
# 教学等价伪代码，不是 SGLang API。
state = {pooled_index_cache: [], tail: []}       # owner: DSATokenToKVPool
for new_token in request:
    key, gate = make_index_key_and_gate(new_token)
    state.tail.append((key, gate))               # 写未完整 pool
    if len(state.tail) == index_kpool:
        pooled_key = learned_compress(state.tail)
        state.pooled_index_cache.append(pooled_key)  # 写完整 pool
        state.tail.clear()

    pool_scores = score(query, state.pooled_index_cache)
    selected_pool_ids = topk(pool_scores, index_topk / index_kpool)
    token_indices = expand_to_raw_tokens(selected_pool_ids)
    output = sparse_mla(token_indices + visible_tail_indices, main_kv)
~~~

真实 decode 写入由 [`DSATokenToKVPool.kpool_decode_update_index_cache()`](../../python/sglang/srt/mem_cache/memory_pool.py#4987-5023) 转入 KPool kernel；未完整状态的 owner 创建于 [`_init_kpool_compress_tail_buffers()`](../../python/sglang/srt/mem_cache/memory_pool.py#4909-4959)。PyTorch reference 没有请求级持久 cache，[`_pool_keys()`](../../scripts/tiny_glm5_flash.py#411-447) 会在每次整段 `forward` 里重建 pool；这是参考实现和 SGLang 增量 runtime 的主要生命周期差异。

`page_size=64` 是当前 CUDA DSA 存储/kernel 约束，不是 GLM 模型理论参数。当前 [`IndexerKPool.__init__`](../../python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py#92-108) 还要求 `64 % index_kpool == 0`；当 `index_kpool=4` 时，一个 main KV page 中正好有 16 个逻辑 KPool，不会让一个 4-token group 跨 page。[`build_pooled_page_table_64()`](../../python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py#16-28) 从原 page table 每隔 4 列取一列，只派生 packed index 视图，不申请新 page。

`IndexerKPool.__init__` 的当前入口位置是 [`dsa_indexer_kpool.py#56-165`](../../python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py#56-165)；Ampere 分支在该 pool 内使用 BF16 index rows。

## 6. GPU backend：SM120 与 Ampere（SM80/SM86）

### 6.1 已验证的 SM120 路径

RTX 5060 Ti 的 SM120 smoke 使用：

~~~text
KDA: Triton
DSA KV cache: FP8 E4M3
DSA prefill/decode: flashinfer_sparse_mla
tiny formal sparse-MLA shape: eager Torch fallback
tiny pooled score top-k: 小 group_topk 走 eager Torch fallback
MoE: SGLang default Triton fallback
mHC: eager path
~~~

`flashinfer_sparse_mla_forward()` 在 [`flash_mla_sm120.py`](../../python/sglang/kernels/ops/attention/flash_mla_sm120.py#687-821) 中先尝试 FlashInfer；如果 kernel dispatch table 没有 tiny shape，就进入 `_torch_sparse_mla_forward()`，直接解码 DSA packed cache 并做 selected-token attention。

### 6.2 A100/SM80 与 SM86 适配路径（脚本分支选择）

SM80（A100）和 SM86 都没有当前 SM120 的 GLM FP8 sparse-MLA 路径，因此 smoke
脚本会根据 `torch.cuda.get_device_capability()` 分别进入 A100/SM80 分支或 SM86
分支；两条分支当前都使用 BF16 + FA3/Torch 组合，不需要手工改源码。A100 分支
单独保留，后续可以在不影响 SM86 的情况下替换 A100 专用 kernel。

两条 Ampere 分支对应的参数是：

~~~python
kv_cache_dtype="bfloat16"
dsa_prefill_backend="fa3"
dsa_decode_backend="fa3"
dsa_paged_mqa_logits_backend="torch"
dsa_topk_backend="torch"
~~~

其中主 MLA cache 和 DSA index cache 都是 BF16；Ampere 仍复用分页 byte facade，
但每个 index row 按 BF16 128 维存储，不调用 FP8 quant/store kernel。logits
改用 [`torch_mqa_logits`](../../python/sglang/srt/layers/attention/dsa/torch_mqa_logits.py) 计算。Ampere 需要单独处理：

1. 主 KV cache 改为 BF16，避免依赖 SM120 FP8 sparse layout；
2. DSA indexer 不依赖 FP8 或 DeepGEMM paged-MQA logits，改成 BF16 + Torch eager index score；
3. selected-token sparse attention 使用 FA3 的 `only_qv` 路径，FA3 当前支持 SM86；
4. KDA 继续使用 Triton；
5. mHC 和 MoE 继续走 eager/default fallback。

新增的 [`DSAPagedMQALogitsBackend.TORCH`](../../python/sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py#11-61) 在 `auto` 模式下也会为 SM80/SM86 选择 Torch；显式指定 `deepgemm` 仍保持严格行为。Ampere 的 pool 会把 index cache 物理 stride 从 FP8+scale 的 132 bytes/row 切换为 BF16 的 256 bytes/row，SM120 tiny fallback 仍读取 FP8 packed cache，不能混用。换句话说：

~~~text
reference CUDA forward       ✅ SM86 可直接尝试
SGLang runner（自动识别）     ✅ SM120 / SM86
SM86 DSA indexer logits       ✅ Torch eager
SM86 sparse attention         ✅ FA3 only_qv
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

### 7.2 SGLang smoke（SM120 / SM86）

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
  --page-size 64 \
  --dtype bfloat16
~~~

`pytorch_model.bin` 供 reference state-dict reload 验证；SGLang smoke 使用 `load_format="dummy"`，不会误把 reference state dict 当成生产 GLM checkpoint。

同一条 SGLang 命令可在 SM86 上运行。脚本会自动使用
`kv_cache_dtype=bfloat16`、`dsa_prefill/decode_backend=fa3`、
`dsa_paged_mqa_logits_backend=torch` 和 `dsa_topk_backend=torch`；SM120 则保留
`flashinfer_sparse_mla + fp8_e4m3`。因此换到 RTX 30/Ampere 机器时无需改命令：

~~~bash
PYTHONPATH=python python scripts/tiny_glm5_flash_sglang.py \
  --model-dir /tmp/tiny-glm5-flash-sglang \
  --batch-size 2 \
  --seq-len 8 \
  --page-size 64 \
  --dtype bfloat16
~~~

上面的 $L=8$ 只验证 all-visible 路径。要在 SGLang 中真正观察“多于 2 个完整 pool，只选 2 个”，使用当前 runner 的 68-token 调试规模：

~~~bash
PYTHONPATH=python python scripts/tiny_glm5_flash_sglang.py \
  --model-dir /tmp/tiny-glm5-flash-sglang \
  --batch-size 2 \
  --seq-len 68 \
  --page-size 64 \
  --dtype bfloat16
~~~

$L=68$ 跨越两个 main KV page，最后一个 prefill query 可见 17 个完整 pool，但 `group_topk=8/4=2`，因此 pooled indexer 会走小 bucket 的 Torch eager top-k，展开后只留 8 个原始 token 位置给 formal sparse MLA。当前工作区已实测 batch 2 的该命令能完成 prefill 和 1-token decode，输出 `SGLang CUDA forward: ok`。

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
- PyTorch reference 直接使用 `torch.topk`，能在 $L\ge12$ 演示 tiny `group_topk=2` 的真正 pool 裁剪；
- SGLang 的 [`topk_from_pooled_history_logits()`](../../python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py#537-655) 对 `128/160/192/224/256/512/2048` 保留已编译快路径；不在集合中的 tiny $group\_topk=2$ 改走 `DSATopKBackend.TORCH.topk_func`；
- $L=8$ smoke 仍只能证明 all-visible 路径；$L=68$ smoke 和 [`test_small_group_topk_uses_torch_fallback`](../../test/registered/kernels/test_dsa_kpool_multi_pool.py#46-74) 才覆盖真正 pool score top-k；
- 小 `group_topk` fallback 要求 CUDA `float32 logits`，且不支持只在 fused fast path 才使用的 `page_table_row_index`；
- SM120 eager sparse fallback 是可读参考实现，不是 GLM-5.3-Flash 的真实 kernel；
- SM86 的 SGLang DSA runner 已按 BF16 + Torch/FA3 路径适配；未在当前无 Ampere GPU 的环境中做真实设备 smoke；
- 当前没有把 tiny model 注册成正式的 GLM-5 模型，也没有修改原始 GLM registry。

如果继续完善 SM86，优先应在真实 A100/SM86 机器上做性能和数值 smoke，重点观察 DSA backend 的三层：`indexer score`、`paged KV gather`、`selected-token attention`。KDA、mHC、MoE 和四层 schedule 不需要再复制一套模型结构。
