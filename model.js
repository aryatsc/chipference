// Roofline-Pareto model of MegaLlama-3T decode on a given chip.
// Pure functions, no DOM. Works in the browser (window.Model) and in node (require).
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------------
  // MegaLlama-3T: Llama-3 architecture scaled up (GQA-8, head_dim 128, SwiGLU).
  // ---------------------------------------------------------------------------
  const MODEL = {
    name: 'MegaLlama-3T',
    layers: 240,
    d: 32768,
    nHeads: 256,
    nKvHeads: 8,
    headDim: 128,
    ffn: 106496,          // 3.25 * d, same ratio as Llama-3-405B
    vocab: 128256,
  };
  MODEL.attnParamsPerLayer = 2 * MODEL.d * MODEL.d + 2 * MODEL.d * MODEL.nKvHeads * MODEL.headDim;
  MODEL.ffnParamsPerLayer = 3 * MODEL.d * MODEL.ffn;
  MODEL.embedParams = MODEL.vocab * MODEL.d; // untied embedding and LM head, each this size
  MODEL.totalParams = MODEL.layers * (MODEL.attnParamsPerLayer + MODEL.ffnParamsPerLayer) + 2 * MODEL.embedParams;

  // ---------------------------------------------------------------------------
  // Achievable: what a best-in-class implementation can reach. No blanket "x% of peak" factors;
  // the gap to speed of light comes from three physical effects, the same for every chip:
  //   1. sustained rates: HBM can't stream at its spec rate (refresh, bank and bus turnaround), and
  //      tensor cores can't hold boost clock at the power limit under sustained GEMM load
  //   2. a fixed sync cost per dependent kernel (launch, pipeline fill, tail), even with
  //      CUDA graphs + programmatic dependent launch; norms, RoPE and activations are fused away
  //   3. decode GEMMs pad the token dimension to the MMA instruction's minimum width
  // Link latencies apply to both achievable and speed of light: they are physics, not software.
  // ---------------------------------------------------------------------------
  const ASSUME = {
    weightBytes: 0.5625,  // NVFP4: 4-bit values + one FP8 scale per 16 values = 4.5 bits
    kvBytes: 1,           // FP8 KV cache
    actBytes: 2,          // BF16 activations (GEMM inputs/outputs, collectives, PP sends)
    hbmSustained: 0.90,   // sustained / spec HBM bandwidth for large streams (B200 microbenchmarks: ~82-90%)
    clockSustained: 0.80, // sustained / boost tensor clock under power-limited GEMM load (estimate)
    linkEff: 1.0,         // NVLink SHARP all-reduce reaches ~100% of spec bus bandwidth on NVL72
    kernelSyncUs: 1.0,    // fixed cost per dependent kernel (launch + fill + tail) with CUDA graphs + PDL
    kernelsPerLayer: 5,   // QKV, attention, O proj, gate/up, down (norm, RoPE, SiLU fused)
    mmaPad: 16,           // decode GEMM token dim padded to the MMA minimum (swap-AB kernels)
    nvlCollectiveUs: 3.0, // small-message all-reduce in NVLink domain (8xB200 nccl ~3 us, low-latency kernel ~2.4 us)
    nvlP2pUs: 2.0,        // one point-to-point send over NVLink
    netP2pUs: 5.0,        // one GPU-to-GPU send over the scale-out network (rough; no public CX-8 figure)
    hbmUsable: 0.92,      // fraction of HBM capacity usable for weights + KV
    gpuIdle: 0.20,        // GPU draw while waiting (leakage, HBM refresh, clocks) as a share of its rated power
    ppOptions: [1, 2, 3, 4, 6, 8],
  };

  // Speed of light: spec rates, no kernel sync cost, no padding. Link latencies stay.
  const SOL = Object.assign({}, ASSUME, { hbmSustained: 1, clockSustained: 1, linkEff: 1, kernelSyncUs: 0, mmaPad: 1 });

  const ceilTo = (x, q) => Math.ceil(x / q) * q;

  function divisors(n) {
    const out = [];
    for (let i = 1; i <= n; i++) if (n % i === 0) out.push(i);
    return out;
  }

  // Parallel configurations for one decode replica.
  //   t: tensor-parallel width of each pipeline stage (inside the NVLink domain)
  //   a: attention TP width (a < t means attention is data-parallel over t/a groups)
  //   p: pipeline stages
  function configs(chip, strategies) {
    const out = [];
    const domain = Math.max(1, Math.round(chip.domain));
    const pps = strategies.pp ? ASSUME.ppOptions : [1];
    for (const t of divisors(domain)) {
      const attnWidths = new Set([t]);
      if (strategies.dpAttn) for (const a of [1, 2, 4, 8]) if (a < t && t % a === 0) attnWidths.add(a);
      for (const a of attnWidths) for (const p of pps) out.push({ t, a, p });
    }
    return out;
  }

  // Per-chip quantities that don't depend on batch.
  function prepare(chip, cfg, work, A) {
    const M = MODEL;
    const { t, a, p } = cfg;
    const g = t / a;                                   // attention DP groups per stage
    const hq = Math.ceil(M.nHeads / a);                // query heads per chip
    const hkv = Math.max(1, Math.ceil(M.nKvHeads / a)); // KV heads per chip (replicated when a > 8)
    const layersPerStage = Math.ceil(M.layers / p);

    const qkvParams = M.d * (hq + 2 * hkv) * M.headDim;
    const oParams = hq * M.headDim * M.d;
    const ffnParams = M.ffnParamsPerLayer / t;
    const lmParams = M.embedParams / t;

    const hbmBW = chip.hbmTBs * 1e12 * A.hbmSustained;
    const fp4 = chip.fp4 * 1e15 * A.clockSustained;
    const fp8 = chip.fp8 * 1e15 * A.clockSustained;
    const nvlBW = chip.nvlTBs / 2 * 1e12 * A.linkEff;     // spec is bidirectional; use one direction
    const netBW = chip.nicGbps / 8 * 1e9 * A.linkEff;

    const ctxAvg = work.isl + work.osl / 2;
    const ctxMax = work.isl + work.osl;

    // Memory: worst stage holds its layers plus one embedding/LM-head shard (two when p = 1).
    const weightBytes = A.weightBytes * (layersPerStage * (qkvParams + oParams + ffnParams) + (p === 1 ? 2 : 1) * lmParams);
    const kvBytesPerUserLayer = ctxMax * 2 * hkv * M.headDim * A.kvBytes;
    const kvBytesPerUser = kvBytesPerUserLayer * layersPerStage / g; // users are spread over g groups
    const free = chip.hbmGB * 1e9 * A.hbmUsable - weightBytes;
    const bMax = free > 0 ? free / kvBytesPerUser : 0;
    const bMin = g * p; // every DP group in every microbatch needs at least one sequence

    const chips = t * p;
    const ppOverNet = chips > domain(chip);

    return { A, cfg, g, hq, hkv, layersPerStage, qkvParams, oParams, ffnParams, lmParams,
      hbmBW, fp4, fp8, nvlBW, netBW, ctxAvg, weightBytes, bMax, bMin, chips, ppOverNet };
  }

  function domain(chip) { return Math.max(1, Math.round(chip.domain)); }

  // Roofline time of one op: max(memory, compute). Returns [time, isComputeBound].
  function op(bytes, flops, bw, peak) {
    const tm = bytes / bw, tc = flops / peak;
    return tc > tm ? [tc, true] : [tm, false];
  }

  // Time per output token (s) for replica batch B, with a breakdown of where time goes.
  function decodeStep(P, B) {
    const M = MODEL, A = P.A;
    const { t, p } = P.cfg;
    const m = B / p;               // tokens per microbatch, per stage
    const mAttn = m / P.g;         // tokens per attention DP group
    const mAttnPad = ceilTo(mAttn, A.mmaPad);
    const mPad = ceilTo(m, A.mmaPad);
    const br = { weights: 0, kv: 0, compute: 0, comm: 0, overhead: 0 };
    const add = ([time, cb], memKey) => { br[cb ? 'compute' : memKey] += time; };

    // --- one layer on one chip -------------------------------------------------
    const layer = { weights: 0, kv: 0, compute: 0, comm: 0, overhead: 0 };
    const addL = ([time, cb], memKey) => { layer[cb ? 'compute' : memKey] += time; };
    // QKV projection
    addL(op(P.qkvParams * A.weightBytes + mAttn * (M.d + (P.hq + 2 * P.hkv) * M.headDim) * A.actBytes,
            2 * mAttnPad * P.qkvParams, P.hbmBW, P.fp4), 'weights');
    // Attention over the KV cache (FP8 math, QK^T and PV)
    addL(op(mAttn * P.ctxAvg * 2 * P.hkv * M.headDim * A.kvBytes,
            4 * mAttn * P.ctxAvg * P.hq * M.headDim, P.hbmBW, P.fp8), 'kv');
    // Output projection
    addL(op(P.oParams * A.weightBytes + mAttn * (P.hq * M.headDim + M.d) * A.actBytes,
            2 * mAttnPad * P.oParams, P.hbmBW, P.fp4), 'weights');
    // SwiGLU FFN (gate+up, down), TP over all t chips, all m tokens
    addL(op(P.ffnParams * A.weightBytes + m * (2 * M.d + 3 * M.ffn / t) * A.actBytes,
            2 * mPad * P.ffnParams, P.hbmBW, P.fp4), 'weights');
    // Two collectives per layer (after attention, after FFN), on the critical path.
    if (t > 1) {
      const msg = m * M.d * A.actBytes;
      layer.comm += 2 * (A.nvlCollectiveUs * 1e-6 + 2 * (t - 1) / t * msg / P.nvlBW);
    }
    layer.overhead += A.kernelsPerLayer * A.kernelSyncUs * 1e-6;

    for (const k in br) br[k] += layer[k] * M.layers; // every token visits every layer once

    // LM head on the last stage
    add(op(P.lmParams * A.weightBytes + m * M.d * A.actBytes, 2 * mPad * P.lmParams, P.hbmBW, P.fp4), 'weights');
    br.overhead += A.kernelSyncUs * 1e-6;

    // Pipeline hops: p sends per token step (incl. the sampled token back to stage 0)
    if (p > 1) {
      const msg = m * M.d * A.actBytes;
      const hop = P.ppOverNet ? A.netP2pUs * 1e-6 + msg / P.netBW : A.nvlP2pUs * 1e-6 + msg / P.nvlBW;
      br.comm += p * hop;
    }

    // Each microbatch passes every stage once per token, so per-token latency is the sum over
    // all layers (computed above with m tokens). With p microbatches in flight, every stage is busy.
    const tpot = br.weights + br.kv + br.compute + br.comm + br.overhead;
    return { tpot, br };
  }

  // Prefill cost in chip-seconds per request (compute-bound, comms overlapped).
  function prefillChipSeconds(chip, work, A) {
    const M = MODEL;
    const linear = 2 * (M.totalParams - 2 * M.embedParams) * work.isl;
    const attn = 2 * M.layers * work.isl * work.isl * M.nHeads * M.headDim; // causal QK^T + PV
    return linear / (chip.fp4 * 1e15 * A.clockSustained) + attn / (chip.fp8 * 1e15 * A.clockSustained);
  }

  // Point for a given config and batch.
  function point(chip, P, B, work, prefillCS) {
    const { tpot, br } = decodeStep(P, B);
    const A = P.A;
    const decodeThr = B / (tpot * P.chips);            // output tok/s per decode chip
    // Power tracks utilization: the static share is always drawn; the rest only while the GPU is
    // moving data or computing, not while it waits on collectives, sends or kernel sync.
    const busy = (br.weights + br.kv + br.compute) / tpot;
    // Non-GPU overhead (CPUs, NICs, switches, cooling share) is always drawn; the GPU draws its idle
    // share always and the rest only while busy.
    const allIn = chip.gpuW + chip.otherW;
    const decodeW = chip.otherW + chip.gpuW * (A.gpuIdle + (1 - A.gpuIdle) * busy);
    // Per request: OSL tokens of decode GPU-time plus the prefill GPU-time (prefill GPUs run flat out).
    const decodeGpuS = work.osl / decodeThr, prefillGpuS = work.includePrefill ? prefillCS : 0;
    const thr = work.osl / (decodeGpuS + prefillGpuS);
    const joules = decodeGpuS * decodeW + prefillGpuS * allIn;
    return {
      interactivity: 1 / tpot,
      usersPerChip: B / P.chips,
      thr,
      thrPerKW: work.osl / joules * 1000,               // output tok/s per kW of average draw
      busy, decodeW,
      tpot, br, B, cfg: P.cfg, chips: P.chips, ppOverNet: P.ppOverNet,
    };
  }

  // Largest B in [bMin, bMax] with tpot(B) <= target. Null if infeasible.
  function maxBatchFor(P, target) {
    if (P.bMax < P.bMin) return null;
    if (decodeStep(P, P.bMin).tpot > target) return null;
    if (decodeStep(P, P.bMax).tpot <= target) return P.bMax;
    let lo = P.bMin, hi = P.bMax;
    for (let i = 0; i < 40; i++) {
      const mid = Math.sqrt(lo * hi);
      if (decodeStep(P, mid).tpot <= target) lo = mid; else hi = mid;
    }
    return lo;
  }

  // Pareto frontier for one chip.
  //   mode 'interactivity': best throughput at interactivity >= x, for each x in grid
  //   mode 'concurrency':   best throughput at exactly x users per decode chip
  function frontier(chip, work, strategies, mode, grid, metric, sol) {
    const A = sol ? SOL : ASSUME;
    const preps = configs(chip, strategies).map(c => prepare(chip, c, work, A)).filter(P => P.bMax >= P.bMin);
    const prefillCS = prefillChipSeconds(chip, work, A);
    const val = pt => metric === 'perKW' ? pt.thrPerKW : pt.thr;
    return grid.map(x => {
      let best = null;
      for (const P of preps) {
        let B;
        if (mode === 'interactivity') B = maxBatchFor(P, 1 / x);
        else { B = x * P.chips; if (B < P.bMin || B > P.bMax) B = null; }
        if (B == null) continue;
        const pt = point(chip, P, B, work, prefillCS);
        if (!best || val(pt) > val(best)) best = pt;
      }
      return best ? Object.assign(best, { x, y: val(best) }) : null;
    });
  }

  // Highest interactivity any config reaches (B = bMin).
  function maxInteractivity(chip, work, strategies, sol) {
    const A = sol ? SOL : ASSUME;
    let best = 0;
    for (const c of configs(chip, strategies)) {
      const P = prepare(chip, c, work, A);
      if (P.bMax < P.bMin) continue;
      best = Math.max(best, 1 / decodeStep(P, P.bMin).tpot);
    }
    return best;
  }

  const api = { MODEL, ASSUME, SOL, configs, prepare, decodeStep, prefillChipSeconds, frontier, maxInteractivity, point };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Model = api;
})(this);
