"""Day08 的 Mamba-2 推理教学模型：单序列、一个 B/C group、CPU 可运行。

随机教学权重，不加载 Falcon-H1，不实现训练、TP、量化、混合 batch 或 GPU kernel。
同时实现逐 token recurrence 与 chunk 内矩阵计算，用数值对照验证计算和状态闭环。
"""

import torch
from torch import nn
from torch.nn import functional as F


def clone_state(state):
    """完整复制 conv 和 temporal；destination 可以独立继续写。"""
    return {name: value.clone() for name, value in state.items()}


def ssm_chunked(x, dt, A, B, C, D, initial, chunk_size):
    """教学 SSD：x=[T,H,P], dt=[T,H], B/C=[T,N], initial=[H,P,N]。

    dt 已做 softplus；A=[H] 为负数。只在 chunk 边界传递 state。
    chunk 内使用 [H,Q,Q] 因果权重矩阵；不是生产用的高效 GPU 实现。
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    state = initial.clone()
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        xc = x[start : start + chunk_size]
        dc = dt[start : start + chunk_size]
        bc = B[start : start + chunk_size]
        cc = C[start : start + chunk_size]
        q = xc.shape[0]
        prefix = (dc * A).cumsum(dim=0)  # [Q,H]，log 衰减的前缀和
        causal = torch.ones(q, q, dtype=torch.bool, device=x.device).tril()
        # 每列 s 独立累加 s+1..t，避免两段很大的前缀和相减。
        terms = (dc * A).T[:, :, None].expand(-1, -1, q)
        segments = terms.masked_fill(~causal.tril(-1), 0).cumsum(dim=1)
        L = segments.masked_fill(~causal, -torch.inf).exp()  # [H,Q,Q]

        # chunk 内部 token 之间的贡献：先算 C_t · B_s，再乘衰减和 dt_s。
        CB = cc @ bc.T  # [Q,Q]；行是读出位置 t，列是写入位置 s
        weights = L * CB[None, :, :] * dc.T[:, None, :]
        within = torch.einsum("hts,shp->thp", weights, xc)
        # 之前所有 chunk 的历史，只通过入口 state 贡献给当前输出。
        from_previous = prefix.exp()[:, :, None] * torch.einsum(
            "hpn,tn->thp", state, cc
        )
        outputs.append(within + from_previous + D[None, :, None] * xc)

        # 只提交这个 chunk 的末态，供下一个 chunk 继续使用。
        tail_weight = L[:, -1, :].T * dc
        written = torch.einsum("th,thp,tn->hpn", tail_weight, xc, bc)
        state = prefix[-1].exp()[:, None, None] * state + written
    return torch.cat(outputs, dim=0), state


class TinyMamba2(nn.Module):
    """单个 Mamba-2 教学 mixer；state 由调用者持有，权重由模块持有。

    ``use_rms_norm=False`` 对齐 Falcon-H1 当前默认配置；打开它可观察
    ``Mixer2RMSNormGated`` 的可选 gated RMSNorm 分支。
    """

    def __init__(
        self,
        d_model=6,
        heads=2,
        head_dim=2,
        d_state=3,
        kernel=3,
        use_rms_norm=False,
    ):
        super().__init__()
        assert kernel >= 2
        self.H, self.P, self.N, self.K = heads, head_dim, d_state, kernel
        self.use_rms_norm = use_rms_norm
        self.inner = heads * head_dim
        self.conv_dim = self.inner + 2 * d_state  # 一个 B/C group
        self.in_proj = nn.Linear(d_model, self.inner + self.conv_dim + heads, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(self.conv_dim, 1, kernel) * 0.2)
        self.conv_bias = nn.Parameter(torch.zeros(self.conv_dim))
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))
        self.D = nn.Parameter(torch.ones(heads))
        if self.use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.inner))
        else:
            self.register_parameter("norm_weight", None)
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def empty_state(self):
        # 与权重同 device/dtype；教学代码使用统一 dtype。
        return {
            "conv": self.A_log.new_zeros(self.conv_dim, self.K - 1),
            "temporal": self.A_log.new_zeros(self.H, self.P, self.N),
        }

    def project(self, hidden):
        # hidden 可以是 [D_model]，也可以是 [T,D_model]。
        projected = self.in_proj(hidden)
        return torch.split(projected, [self.inner, self.conv_dim, self.H], dim=-1)

    def finish(self, y, gate):
        # TP=1、一个 norm group；Falcon-H1 默认可关闭 RMSNorm。
        v = y.flatten(start_dim=-2) * F.silu(gate)
        if self.use_rms_norm:
            # 与 Mixer2RMSNormGated.forward_* 的 norm_before_gate=False 对齐。
            v = v * torch.rsqrt(v.square().mean(dim=-1, keepdim=True) + 1e-6)
            v = v * self.norm_weight
        return self.out_proj(v)

    def step(self, hidden, state):
        # 单 token：[D_model] + 旧 state -> [D_model] + 原地更新的 state。
        gate, raw_u, raw_dt = self.project(hidden)
        window = torch.cat([state["conv"], raw_u[:, None]], dim=-1)
        u = F.silu((window * self.conv_weight[:, 0]).sum(-1) + self.conv_bias)
        state["conv"].copy_(window[:, 1:])  # 保存 raw 窗口，不保存 u
        x, B, C = torch.split(u, [self.inner, self.N, self.N], dim=-1)
        x = x.view(self.H, self.P)
        dt = F.softplus(raw_dt + self.dt_bias)
        A = -self.A_log.exp()
        new_s = (
            torch.exp(dt * A)[:, None, None] * state["temporal"]
            + dt[:, None, None] * x[:, :, None] * B[None, None, :]
        )
        state["temporal"].copy_(new_s)
        y = torch.einsum("hpn,n->hp", new_s, C) + self.D[:, None] * x
        return self.finish(y, gate), state

    def recurrent(self, hidden, state):
        outputs = []
        for h_t in hidden:
            y_t, state = self.step(h_t, state)
            outputs.append(y_t)
        return torch.stack(outputs), state

    def prefill(self, hidden, state, chunk_size=4):
        # 多 token；投影、卷积并行计算，SSM 使用 chunked SSD。
        gate, raw_u, raw_dt = self.project(hidden)
        history = torch.cat([state["conv"], raw_u.T], dim=-1)
        u = F.silu(F.conv1d(
            history[None], self.conv_weight, self.conv_bias,
            groups=self.conv_dim,
        )[0].T)
        state["conv"].copy_(history[:, -(self.K - 1):])
        x, B, C = torch.split(u, [self.inner, self.N, self.N], dim=-1)
        dt = F.softplus(raw_dt + self.dt_bias)
        y, last_s = ssm_chunked(
            x.view(-1, self.H, self.P), dt, -self.A_log.exp(), B, C,
            self.D, state["temporal"], chunk_size,
        )
        state["temporal"].copy_(last_s)
        return self.finish(y, gate), state


@torch.inference_mode()
def demonstrate():
    torch.manual_seed(8)
    # 采用 CPU float64，便于把教学算法差异与低精度舍入区分开。
    model = TinyMamba2().double().eval()
    hidden = torch.randn(7, 6, dtype=torch.float64)
    worst = 0.0
    for nonzero_initial in (False, True):
        initial = model.empty_state()
        if nonzero_initial:
            for value in initial.values():
                value.copy_(torch.randn_like(value) * 0.1)
        reference, ref_state = model.recurrent(hidden, clone_state(initial))
        # 覆盖 chunk=1、多 chunk、非整除尾段和整个序列小于 chunk。
        for chunk_size in (1, 2, 4, 8):
            actual, actual_state = model.prefill(hidden, clone_state(initial), chunk_size)
            torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
            for name in initial:
                torch.testing.assert_close(actual_state[name], ref_state[name],
                                           rtol=1e-10, atol=1e-10)
            worst = max(worst, (actual - reference).abs().max().item())

    # prefix 分叉：同一输入分两次 prefill + decode，应与一次处理一致。
    prefix_out, prefix = model.prefill(hidden[:3], model.empty_state(), chunk_size=2)
    pool = {7: clone_state(prefix), 8: clone_state(prefix)}
    suffix_out, _ = model.prefill(hidden[3:6], pool[8], chunk_size=2)
    ptrs = {name: value.data_ptr() for name, value in pool[8].items()}
    last_out, _ = model.step(hidden[6], pool[8])
    full_out, full_state = model.recurrent(hidden, model.empty_state())
    resumed = torch.cat([prefix_out, suffix_out, last_out[None]])
    torch.testing.assert_close(resumed, full_out, rtol=1e-10, atol=1e-10)
    for name in prefix:
        assert ptrs[name] == pool[8][name].data_ptr()
        assert pool[8][name].data_ptr() != pool[7][name].data_ptr()
        assert torch.equal(pool[7][name], prefix[name])
        torch.testing.assert_close(pool[8][name], full_state[name], rtol=1e-10, atol=1e-10)
    assert not torch.equal(pool[8]["temporal"], prefix["temporal"])
    print(f"PASS: recurrent vs chunked prefill; max output error = {worst:.3e}")
    print("PASS: nonzero initial state, chunk tails, prefix continuation, COW, in-place decode")
    print("output:", tuple(full_out.shape), "conv:", tuple(full_state["conv"].shape),
          "temporal:", tuple(full_state["temporal"].shape))


if __name__ == "__main__":
    demonstrate()
