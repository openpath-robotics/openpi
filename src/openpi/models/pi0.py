import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _rtc_soft_mask(d, s, H: int) -> jax.Array:
    """Soft mask W for RTC guidance (paper Eq.5).

    Equivalent to kinetix ``get_prefix_weights(start=d, end=H-s, total=H, schedule="exp")``.

    Regions:
      i < d       : W = 1.0   (frozen prefix — will execute during inference delay)
      d <= i < H-s: W = exp-decaying 1→0  (intermediate — soft guidance)
      i >= H-s    : W = 0.0   (fresh region — no constraint from previous chunk)

    Args:
        d: inference_delay.  ``start`` in kinetix notation.
        s: execution_horizon.  ``end = H - s`` in kinetix notation.
        H: action_horizon (static Python int).

    Example (d=2, s=4, H=10):
        i :  0    1    2    3    4    5    6    7    8    9
        W :  1.0  1.0  *    *    *    *    0.0  0.0  0.0  0.0
                        ^              ^
                       d=2           H-s=6
        (* = exp schedule, decaying from ~1 to ~0)
    """
    # Convert to float32 JAX arrays so d and s can be dynamic (no recompile per value).
    d_f = jnp.asarray(d, dtype=jnp.float32)
    s_f = jnp.asarray(s, dtype=jnp.float32)
    H_f = float(H)

    # kinetix line 52: start = jnp.minimum(start, end)
    # Guard: if d > H-s (invalid params), clamp d_eff to H-s.
    d_eff = jnp.minimum(d_f, H_f - s_f)

    i = jnp.arange(H, dtype=jnp.float32)

    # Base linear weight c_i (paper Eq.5 denominator: H-s-d+1).
    # kinetix: clip((start-1-i)/(end-start+1)+1, 0, 1)
    #        = clip((H-s-i)/(H-s-d_eff+1), 0, 1)   [algebra: (d_eff-1-i)/(H-s-d_eff+1)+1 = (H-s-i)/(H-s-d_eff+1)]
    # For i < d_eff: (H-s-i)/(H-s-d_eff+1) >= 1, clipped to 1.
    # For i in [d_eff, H-s): decays from <1 toward 0.
    denom = jnp.maximum(H_f - s_f - d_eff + 1.0, 1.0)
    c = jnp.clip((H_f - s_f - i) / denom, 0.0, 1.0)

    # Exp schedule (kinetix line 60): w = c * expm1(c) / (e - 1)
    # At c=1 (frozen region): expm1(1)/(e-1) = 1.0  → W=1 ✓
    # At c=0 (boundary):      w = 0                 → W=0 ✓
    w = c * jnp.expm1(c) / (jnp.e - 1)

    # Trailing zeros: W=0 for i >= H-s  (kinetix: where(arange >= end, 0, w))
    return jnp.where(i >= H_f - s_f, 0.0, w)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # Pass remat_policy through to gemma configs (supports 16GB VRAM variants).
        remat_policy = getattr(config, "remat_policy", "nothing_saveable")
        paligemma_config.remat_policy = remat_policy
        action_expert_config.remat_policy = remat_policy
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if config.wrench_dim > 0:
            self.wrench_proj = nnx.Linear(config.wrench_dim, action_expert_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

            # add wrench token (same causal group as state: ar_mask=False)
            if obs.wrench is not None:
                wrench_token = self.wrench_proj(obs.wrench)[:, None, :]
                tokens.append(wrench_token)
                input_mask.append(jnp.ones((obs.wrench.shape[0], 1), dtype=jnp.bool_))
                ar_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        prev_action_chunk: at.Float[at.Array, "b ah ad"],
        inference_delay: int | at.Int[at.Array, ""],
        execution_horizon: int | at.Int[at.Array, ""],
        max_guidance_weight: float = 5.0,
    ) -> _model.Actions:
        """Real-Time Chunking (RTC) guided inference — Algorithm 1 GuidedInference.

        Adapted from kinetix ``FlowPolicy.realtime_action`` to pi0's time convention.

        **Time convention mapping**::

            pi0    t: 1 → 0   noise at t=1, clean at t=0,  dt = -1/n  (negative)
            paper  τ: 0 → 1   noise at τ=0, clean at τ=1,  dt = +1/n  (positive)
            relation: τ_paper = 1 - t_pi0

        **Denoiser estimate** (equivalent under τ = 1 - t)::

            pi0:   Â = x_t - t · v_t
            paper: Â = x_τ + (1-τ) · v_τ    (kinetix line 239: x_t + v_t*(1-t))

        **Guided velocity sign** (both produce the same +γ/n·g net displacement)::

            pi0:   v_guided = v_t - γ·g   [MINUS, dt < 0]
            paper: v_guided = v_τ + γ·g   [PLUS,  dt > 0]

        Args:
            prev_action_chunk: A_prev from Algorithm 1, shape (B, H, D).
                ``A_cur[s:]`` right-padded with zeros to length H.
            inference_delay:   d — frozen prefix length (W=1 for i < d).
                Pass as ``jnp.int32`` to avoid JIT recompilation across calls.
            execution_horizon: s — H-s is where W drops to zero (W=0 for i >= H-s).
                Pass as ``jnp.int32`` to avoid JIT recompilation across calls.
            max_guidance_weight: β clip (paper/kinetix default: 5.0).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # pi0 convention: t goes 1 (noise) → 0 (clean), dt is negative.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        H = self.action_horizon

        noise = jax.random.normal(rng, (batch_size, H, self.action_dim))

        # ── Prefix KV cache — computed once, same pattern as sample_actions ───
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        # ── Soft mask W (paper Eq.5 / kinetix get_prefix_weights schedule="exp") ─
        # Mapping to kinetix get_prefix_weights(start, end, total, "exp"):
        #   start = inference_delay          = d
        #   end   = H - execution_horizon    = H - s   (kinetix: prefix_attention_horizon)
        #   total = H
        W = _rtc_soft_mask(inference_delay, execution_horizon, H)  # (H,)

        # ── Suffix forward used inside denoiser (differentiated via jax.vjp) ──
        def _suffix_forward(x_t: jax.Array, t_scalar: jax.Array) -> jax.Array:
            """Suffix-only forward pass reusing the frozen prefix KV cache."""
            t_batch = jnp.broadcast_to(t_scalar, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, t_batch
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # Suffix queries attend to all valid prefix tokens (same as sample_actions).
            prefix_attn_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn_for_suffix, suffix_attn_mask], axis=-1
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -H:])  # (B, H, D)

        # ── RTC guided denoising step (mirrors kinetix realtime_action→step) ──
        def guided_step(carry, _):
            x_t, t = carry

            # Denoiser: Â = x_t - t·v_t   (pi0 convention)
            # kinetix line 239: x_t + v_t*(1-t)  [τ convention, equivalent]
            # has_aux=True: vjp differentiates only the first return value (Â),
            # while v_t is returned as an auxiliary (no gradient flows through it).
            def denoiser(x: jax.Array):
                v = _suffix_forward(x, t)
                return x - t * v, v  # (Â, v_t)

            # VJP — kinetix lines 241, 246:
            #   x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            #   pinv_correction = vjp_fun(error)[0]
            A_hat, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)

            # Weighted error (paper Eq.2 numerator, kinetix line 245):
            #   error = (y - x_1) * weights[:, None]
            # W shape (H,) → broadcast to (B, H, D) via [None, :, None].
            err = (prev_action_chunk - A_hat) * W[None, :, None]

            # Pseudo-inverse correction (kinetix: pinv_correction)
            g = vjp_fn(err)[0]  # (B, H, D)

            # Guidance weight β̃ (paper after Eq.4, kinetix lines 248-250).
            # Variable mapping: t_pi0 ↔ t in pi0,  tau = 1-t = τ_paper.
            #
            # kinetix (τ_paper as variable named t):
            #   inv_r2 = (t² + (1-t)²) / (1-t)²
            #   c      = (1-t) / t
            #
            # pi0 (t_pi0 as variable named t, tau = τ_paper):
            #   inv_r2 = (tau² + t²) / t²       [same formula, swap t↔tau]
            #   c      = t / tau                 [same formula, swap t↔tau]
            tau = 1.0 - t                                              # τ_paper (0→1)
            inv_r2 = (tau ** 2 + t ** 2) / jnp.maximum(t ** 2, 1e-8)
            c = jnp.where(tau > 1e-6, t / tau, max_guidance_weight)   # (1-τ)/τ
            gw = jnp.minimum(c * inv_r2, max_guidance_weight)         # β̃

            # Guided velocity: MINUS for pi0 (PLUS for kinetix, line 251).
            # Both give net per-step correction: +gw/n·g  on x_t.
            v_guided = v_t - gw * g

            return (x_t + dt * v_guided, t + dt), None

        # lax.scan: num_steps iterations, carry=(x_t, t).
        # kinetix line 263: jax.lax.scan(step, (noise, 0.0), length=num_steps)
        # pi0 equivalent: start at t=1.0 (noise), step by dt<0, reach t≈0 (clean).
        (x_0, _), _ = jax.lax.scan(guided_step, (noise, 1.0), length=num_steps)
        return x_0
