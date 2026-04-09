# GPT-2 Ablation Training Algorithm

---

## Algorithm 1 — Learning Rate Schedule

**Algorithm 1:** Cosine Annealing with Linear Warm-Up

**Input:**
- Current step $t$, warm-up steps $T_w$, total steps $T$,
- peak learning rate $\eta_{\max}$, minimum learning rate $\eta_{\min}$

**Output:** Learning rate $\eta_t$

---

```
1  if t < T_w then
2      η_t ← η_max · (t / T_w)                              ▷ Linear warm-up
3  else if t ≥ T then
4      η_t ← η_min                                          ▷ Minimum LR floor
5  else
6      p ← (t - T_w) / (T - T_w)                           ▷ Decay progress p ∈ [0,1]
7      η_t ← η_min + ½(1 + cos(πp)) · (η_max - η_min)
8  end if
9  return η_t
```

---

## Algorithm 2 — Streaming Dataset with Position-Aware Fast Skip

**Algorithm 2:** Token-Buffer Streaming with Resume Skip

**Input:**
- Document stream $\mathcal{D}$, sequence length $L$, tokenizer $\tau$,
- chunks to skip $S$ *(0 for fresh training, $\lfloor N_{\text{seen}} / L \rfloor$ for resumed training)*

**Output:** Sequence pairs $(x, y) \in \mathbb{Z}^L \times \mathbb{Z}^L$

---

```
1   B ← [], s ← 0, skipping ← (S > 0)     ▷ Token buffer, skip counter, skip flag
2   for each document d ∈ D do
3       B += τ(d) || [EOT]                  ▷ Append tokenized document
4       while |B| ≥ L + 1 do
5           c ← B[0 : L+1];  B ← B[L+1 :]  ▷ Slice next chunk
6           if skipping then
7               s ← s + 1
8               if s ≥ S then  skipping ← False  end if
9               continue                    ▷ Discard; no tensor allocation
10          end if
11          x ← c[0:L];  y ← c[1:L+1]
12          yield (x, y)
13      end while
14  end for
```

---

## Algorithm 3 — GPT-2 Pre-Training with Gradient Accumulation and Checkpoint Resumption

**Algorithm 3:** Main Training Loop

**Input:**
- Model $f_\theta$ with parameters $\theta$, dataset $\mathcal{D}$,
- batch size $B$, gradient accumulation steps $G$, total steps $T$,
- $\eta_{\max}$, $\eta_{\min}$, warm-up steps $T_w$, weight decay $\lambda$,
- gradient clip norm $\gamma$, checkpoint path $\rho$ *(optional)*

**Output:** Trained parameters $\theta^*$, training loss sequence $\{\mathcal{L}_t\}$

---

```
1   Initialise f_θ, AdamW optimizer O with (β1, β2) = (0.9, 0.95), λ on ≥2-D params
2   t ← 0;  N ← 0                          ▷ Step counter, total tokens seen
3   if checkpoint ρ is provided then
4       Load θ, O from ρ
5       t ← ρ.t;  N ← ρ.N                  ▷ Restore step; LR schedule continues from t
6   end if
7   S ← ⌊N / L⌋                            ▷ Chunks to skip in dataset stream
8   Initialise data iterator I from Algorithm 2 with skip S
9   while t < T do
10      L_accum ← 0
11      for g = 1 to G do                   ▷ Gradient accumulation
12          (x, y) ← next(I);  x, y ∈ Z^(B × L)
13          ŷ ← f_θ(x)                      ▷ Forward pass with mixed precision
14          ℓ ← (1/G) · L_CE(ŷ, y)
15          ℓ.backward()                    ▷ Accumulate gradients
16          L_accum ← L_accum + ℓ
17          N ← N + B · L
18      end for
19      ‖∇θ‖₂ ← min(‖∇θ‖₂, γ)             ▷ Gradient clipping
20      η_t ← Algorithm 1(t, T_w, T, η_max, η_min)
21      Update θ via O with learning rate η_t
22      t ← t + 1
23      if t mod T_save = 0 then
24          Save checkpoint (θ, O, t, N)
25      end if
26  end while
27  return θ
```

---

## Notation Reference

| Symbol | Description |
|--------|-------------|
| $f_\theta$ | GPT-2 transformer with parameters $\theta$ |
| $t$ | Current training step |
| $T$ | Total training steps |
| $T_w$ | Warm-up steps |
| $T_{\text{save}}$ | Checkpoint save interval |
| $B$ | Micro-batch size |
| $G$ | Gradient accumulation steps (effective batch = $B \times G$) |
| $L$ | Sequence length (context window) |
| $N$ | Total tokens seen so far |
| $S$ | Number of sequence chunks to skip on dataset resume |
| $\eta_t$ | Learning rate at step $t$ |
| $\eta_{\max}, \eta_{\min}$ | Peak and minimum learning rates |
| $\lambda$ | Weight decay coefficient |
| $\gamma$ | Gradient clipping norm threshold |
| $\mathcal{L}_{\text{CE}}$ | Cross-entropy loss |
| $\tau$ | BPE tokenizer |
| $\mathcal{B}$ | Rolling token buffer |
| $\mathcal{D}$ | Streaming document corpus (FineWeb-Edu) |
