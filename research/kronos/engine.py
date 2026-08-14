"""Running Kronos over our bars: hidden states out, and futures sampled from it.

The upstream `KronosPredictor` is not usable for this question, for one reason
worth stating plainly. Its `predict` draws `sample_count` futures and then
**averages them** before returning. The average of many futures is a forecast of
the middle, and we are asking about the tail: how much of tomorrow's
distribution sits below minus two percent. Averaging destroys precisely the
quantity we came for, so the sampling loop is reimplemented here.

The reimplementation is not a reinterpretation. It follows
`auto_regressive_inference` step for step for the `pred_len == 1` case — the
same per-window normalisation, the same clip at five, the same nucleus
filtering, the same trick of decoding a full window with the generated bar
appended and reading only its last position. `verify.py` pins our single-bar
path against theirs on identical inputs, with the randomness removed so the
comparison is an equality rather than a distribution test.

Two savings make this affordable on a laptop, and neither changes the answer:

  * **One transformer pass per day, not one per sample.** Upstream repeats the
    input `sample_count` times *before* the forward pass, which for a one-bar
    horizon recomputes an identical 512-token pass N times over. The first-stage
    logits do not depend on the sample, so we run the body once and draw all N
    samples from the one distribution.
  * **The second stage reuses the first stage's context.** `decode_s2` is cross
    attention whose keys and values are the hidden states; only the query — the
    embedding of the sampled first-stage token — differs between samples. So the
    context is expanded, not recomputed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KRONOS_HOME = Path(os.environ.get("KRONOS_HOME", Path.home() / ".cache/kronos-work/Kronos"))
if str(KRONOS_HOME) not in sys.path:
    sys.path.insert(0, str(KRONOS_HOME))

from model import Kronos, KronosTokenizer  # noqa: E402
from model.kronos import calc_time_stamps, top_k_top_p_filtering  # noqa: E402

#: Their normalisation clamp, in standard deviations. Named because it is easy
#: to mistake for a hyperparameter of ours; it belongs to the pre-training.
CLIP = 5.0

TOKENIZERS = {
    "small": "NeoQuasar/Kronos-Tokenizer-base",
    "base": "NeoQuasar/Kronos-Tokenizer-base",
    "mini": "NeoQuasar/Kronos-Tokenizer-2k",
}
MODELS = {
    "small": "NeoQuasar/Kronos-small",
    "base": "NeoQuasar/Kronos-base",
    "mini": "NeoQuasar/Kronos-mini",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True, slots=True)
class Engine:
    tokenizer: KronosTokenizer
    model: Kronos
    device: str

    @property
    def width(self) -> int:
        return int(self.model.d_model)


def load(size: str = "small", device: str | None = None) -> Engine:
    device = device or pick_device()
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZERS[size]).to(device).eval()
    model = Kronos.from_pretrained(MODELS[size]).to(device).eval()
    return Engine(tokenizer=tokenizer, model=model, device=device)


def normalise(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window standardisation, exactly as `KronosPredictor.predict` does it.

    Each window is scaled by its own mean and standard deviation, which is what
    makes a level-free model of shape possible — and, incidentally, what makes
    the tokens for a given bar depend on which window you are looking at it
    from. There is no leak: the window ends at the decision bar.
    """
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    scaled = (windows - mean) / (std + 1e-5)
    return np.clip(scaled, -CLIP, CLIP), mean, std


def _stamps_for(index: pd.DatetimeIndex, at: np.ndarray, length: int) -> np.ndarray:
    offsets = np.arange(-length + 1, 1)
    flat = pd.Series(index[(at[:, None] + offsets[None, :]).reshape(-1)])
    frame = calc_time_stamps(flat)
    return frame.to_numpy(dtype=np.float32).reshape(at.size, length, -1)


@torch.no_grad()
def hidden_states(
    engine: Engine,
    windows: np.ndarray,
    index: pd.DatetimeIndex,
    at: np.ndarray,
) -> np.ndarray:
    """The model's representation of "here is where the market stands today".

    Returns `[len(at), d_model]` — the final position of the transformer stack,
    which is the state from which Kronos would generate tomorrow. That makes it
    the natural thing to attach a classifier to: everything the model thinks is
    worth remembering about the last 512 days, before it commits to a guess.
    """
    scaled, _, _ = normalise(windows)
    x = torch.from_numpy(scaled.astype(np.float32)).to(engine.device)
    stamp = torch.from_numpy(_stamps_for(index, at, windows.shape[1])).to(engine.device)

    s1, s2 = engine.tokenizer.encode(x, half=True)
    _, context = engine.model.decode_s1(s1, s2, stamp)
    return context[:, -1, :].float().cpu().numpy()


@torch.no_grad()
def tokenize(
    engine: Engine,
    windows: np.ndarray,
    index: pd.DatetimeIndex,
    at: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens and time stamps for a batch of windows, kept on the device."""
    scaled, _, _ = normalise(windows)
    x = torch.from_numpy(scaled.astype(np.float32)).to(engine.device)
    stamp = torch.from_numpy(_stamps_for(index, at, windows.shape[1])).to(engine.device)
    s1, s2 = engine.tokenizer.encode(x, half=True)
    return s1, s2, stamp


def _draw(logits: torch.Tensor, count: int, temperature: float, top_p: float) -> torch.Tensor:
    """`count` draws from one filtered distribution. Shape [batch, count].

    The `top_p >= 1` branch is ours. Upstream's `top_k_top_p_filtering` guards
    both of its bodies (`if top_k > 0`, `if top_p < 1.0`) and has no final
    return, so asking it not to filter hands back `None`. Their own callers
    never notice because their defaults always filter something — but sampling
    the *untruncated* distribution is the whole point here, so we skip the call
    rather than pass it a value it cannot handle.
    """
    scaled = logits / temperature
    filtered = scaled if top_p >= 1.0 else top_k_top_p_filtering(scaled, top_k=0, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, count, replacement=True)


@torch.no_grad()
def second_stage_logits(
    model: Kronos,
    context: torch.Tensor,
    drawn_s1: torch.Tensor,
) -> torch.Tensor:
    """`decode_s2`'s final position, for many candidate first-stage tokens at once.

    Calling `model.decode_s2` once per sample is what a naive implementation
    does, and it is unaffordable: it runs cross attention and a 1024-way vocab
    projection across all 512 positions, of which we read one. The saving is
    structural rather than approximate — the keys and values come from the
    context, which every sample shares, so they are computed once per day and
    only the query differs.

    Written out by hand because their `RotaryPositionalEmbedding` derives its
    positions from the query's length, so handing it a length-one query would
    silently rotate the final bar as if it were the first.

    Args:
        context: `[batch, length, width]` from `decode_s1`.
        drawn_s1: `[batch, count]` candidate first-stage tokens for the last bar.

    Returns:
        `[batch, count, vocab]` second-stage logits for the bar being generated.
    """
    layer = model.dep_layer
    attn = layer.cross_attn
    batch, length, width = context.shape
    count = drawn_s1.size(1)
    heads, head_dim = attn.n_heads, attn.head_dim

    key = attn.k_proj(context).view(batch, length, heads, head_dim).transpose(1, 2)
    value = attn.v_proj(context).view(batch, length, heads, head_dim).transpose(1, 2)

    cos, sin = attn.rotary._update_cos_sin_cache(key, length)
    rotate = attn.rotary._rotate_half
    key = key * cos + rotate(key) * sin

    sibling = model.embedding.emb_s1(drawn_s1)  # [batch, count, width]
    query = attn.q_proj(sibling).view(batch, count, heads, head_dim).transpose(1, 2)
    last_cos, last_sin = cos[:, :, length - 1 : length, :], sin[:, :, length - 1 : length, :]
    query = query * last_cos + rotate(query) * last_sin

    # Not causal: `MultiHeadCrossAttentionWithRoPE` sets `is_causal=self.training`,
    # and we are in eval, so the query sees the whole context. Mirrored here.
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    out = out.transpose(1, 2).contiguous().view(batch, count, width)
    out = attn.out_proj(out)

    fused = layer.norm(context[:, -1:, :] + out)
    return model.head.cond_forward(fused)


def _rotate(attn: torch.nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos + attn.rotary._rotate_half(x) * sin


@torch.no_grad()
def decode_last_bar(
    tokenizer: KronosTokenizer,
    prefix_s1: torch.Tensor,
    prefix_s2: torch.Tensor,
    drawn_s1: torch.Tensor,
    drawn_s2: torch.Tensor,
) -> torch.Tensor:
    """Turn each drawn token pair back into a bar, sharing the work they share.

    Reading a generated token back as a price means running the tokenizer's
    causal decoder over the trailing window. Done naively that is one 512-step
    pass per sample per day, and it dominates everything else in this file by
    two orders of magnitude — the run that motivated writing this was on course
    for about three hours.

    But the samples for a given day differ in exactly one position: the last.
    Causal attention means the 511 positions before it produce identical keys
    and values no matter what was drawn, so they are computed once and the
    samples are a single attention step against that cache.

    This is an exact rearrangement, not an approximation — `verify.py` checks it
    against `tokenizer.decode` and expects floating-point noise. Truncating the
    window instead *would* be an approximation, and a bad one: the same script
    measures it moving predictions by as much as the daily spread.

    Args:
        prefix_s1/prefix_s2: `[batch, length - 1]`, shared by every sample.
        drawn_s1/drawn_s2: `[batch, count]`, the generated final bar.

    Returns:
        `[batch, count, channels]` — the decoded final bar, still normalised.
    """
    prefix_bits = tokenizer.indices_to_bits([prefix_s1, prefix_s2], half=True)
    step_bits = tokenizer.indices_to_bits([drawn_s1, drawn_s2], half=True)
    x = tokenizer.post_quant_embed(prefix_bits)
    step = tokenizer.post_quant_embed(step_bits)

    batch, prefix_len, width = x.shape
    count = step.size(1)
    scale = 1.0

    for layer in tokenizer.decoder:
        attn = layer.self_attn
        heads, head_dim = attn.n_heads, attn.head_dim
        scale = head_dim**-0.5

        # The shared prefix, exactly as `TransformerBlock.forward` would do it.
        h = layer.norm1(x)
        q = attn.q_proj(h).view(batch, prefix_len, heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(batch, prefix_len, heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(batch, prefix_len, heads, head_dim).transpose(1, 2)
        q, k = attn.rotary(q, k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, prefix_len, width)
        x = x + attn.out_proj(out)
        x = x + layer.ffn(layer.norm2(x))

        # The one new position, once per sample, against the cache above. Its
        # rotary position is `prefix_len`, which is why this cannot be done by
        # handing their module a length-one sequence.
        hs = layer.norm1(step)
        qs = attn.q_proj(hs).view(batch, count, heads, head_dim).transpose(1, 2)
        ks = attn.k_proj(hs).view(batch, count, heads, head_dim).transpose(1, 2)
        vs = attn.v_proj(hs).view(batch, count, heads, head_dim).transpose(1, 2)
        cos, sin = attn.rotary._update_cos_sin_cache(qs, prefix_len + 1)
        here_cos, here_sin = cos[:, :, prefix_len:, :], sin[:, :, prefix_len:, :]
        qs = _rotate(attn, qs, here_cos, here_sin)
        ks = _rotate(attn, ks, here_cos, here_sin)

        # Attention over [shared keys, own key]. Splitting the softmax this way
        # avoids materialising a per-sample copy of the whole key cache.
        past = torch.matmul(qs, k.transpose(-2, -1)) * scale  # [b, h, count, prefix]
        mine = (qs * ks).sum(dim=-1, keepdim=True) * scale  # [b, h, count, 1]
        weights = torch.softmax(torch.cat([past, mine], dim=-1), dim=-1)
        blended = torch.matmul(weights[..., :prefix_len], v) + weights[..., prefix_len:] * vs
        blended = blended.transpose(1, 2).contiguous().view(batch, count, width)
        step = step + attn.out_proj(blended)
        step = step + layer.ffn(layer.norm2(step))

    return tokenizer.head(step)


@torch.no_grad()
def sample_next_bar(
    engine: Engine,
    windows: np.ndarray,
    index: pd.DatetimeIndex,
    at: np.ndarray,
    *,
    count: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    decode_tail: int = 512,
    decode_chunk: int = 2048,
    cached: bool = True,
) -> np.ndarray:
    """`[len(at), count]` next-day returns, drawn from the model's own future.

    A return here is `sampled_close / today's actual close - 1`, so the fraction
    of the row at or below -2% is a probability estimate for the very event the
    shipped detector predicts — arrived at with no fitting of any kind.

    `decode_tail` is the one approximation in this file. Turning the generated
    token back into a price runs a causal decoder over the trailing window, and
    doing that for every sample of every day is the dominant cost of the whole
    experiment. Shortening the window changes the answer, in principle, because
    the final position attends to everything before it. How much it changes the
    answer is a measurable quantity rather than a matter of opinion, so
    `verify.py` measures it; the default stays at the full 512 so
    that anyone who does not read this paragraph gets the exact result.
    """
    scaled, mean, std = normalise(windows)
    length = windows.shape[1]
    x = torch.from_numpy(scaled.astype(np.float32)).to(engine.device)
    stamp = torch.from_numpy(_stamps_for(index, at, length)).to(engine.device)

    s1, s2 = engine.tokenizer.encode(x, half=True)
    batch = s1.size(0)

    s1_logits, context = engine.model.decode_s1(s1, s2, stamp)
    drawn_s1 = _draw(s1_logits[:, -1, :], count, temperature, top_p)  # [batch, count]

    s2_logits = second_stage_logits(engine.model, context, drawn_s1)  # [batch, count, vocab]
    drawn_s2 = _draw(s2_logits.reshape(batch * count, -1), 1, temperature, top_p)
    drawn_s2 = drawn_s2.view(batch, count)

    # Upstream appends the generated bar and decodes the trailing window, so the
    # decoder sees the new bar in context. Same here — dropping the oldest token
    # to keep the length — and only the last position is read back.
    tail = min(decode_tail, length)
    prefix_s1 = s1[:, length - tail + 1 :]
    prefix_s2 = s2[:, length - tail + 1 :]

    if cached:
        bar = decode_last_bar(engine.tokenizer, prefix_s1, prefix_s2, drawn_s1, drawn_s2)
        bar = bar.float().cpu().numpy()
    else:
        wide_s1 = prefix_s1.unsqueeze(1).expand(-1, count, -1)
        wide_s2 = prefix_s2.unsqueeze(1).expand(-1, count, -1)
        tail_s1 = torch.cat([wide_s1, drawn_s1.unsqueeze(-1)], dim=2).reshape(batch * count, tail)
        tail_s2 = torch.cat([wide_s2, drawn_s2.unsqueeze(-1)], dim=2).reshape(batch * count, tail)
        pieces = []
        for start in range(0, tail_s1.size(0), decode_chunk):
            stop = start + decode_chunk
            decoded = engine.tokenizer.decode([tail_s1[start:stop], tail_s2[start:stop]], half=True)
            pieces.append(decoded[:, -1, :].float().cpu())
        bar = torch.cat(pieces).numpy().reshape(batch, count, -1)

    # Back to money, with each window's own scale.
    close_at = 3
    restored = bar[:, :, close_at] * (std[:, 0, close_at, None] + 1e-5) + mean[:, 0, close_at, None]
    today = windows[:, -1, close_at][:, None]
    return restored / today - 1.0
