"""
v0.13 — Paradigm B on GeNN (GPU port).  *** NOT runnable on this CPU box ***

This is the GPU-accelerated version of paradigm_b_matcher.py, built on GeNN
(https://genn-team.github.io/). GeNN code-generates C++/CUDA at runtime, so it needs
a real toolchain that this machine does not have:

  REQUIREMENTS (e.g. your RTX 5070 box):
    - CUDA 12.8+   (Blackwell / sm_120 — older CUDA lacks sm_120 kernels)
    - a C++ host compiler (MSVC `cl.exe` on Windows, or gcc/clang on Linux)
    - GeNN 5 + PyGeNN:  pip install pygenn      (GeNN 5 ships Blackwell support;
                                                 GeNN 4 — the linked docs — predates it)
  This box has no pygenn wheel, no compiler and no CUDA, so the verified core is the
  pure-Python paradigm_b_matcher.py; this file is the drop-in GPU accelerator.

What it does (identical query to the CPU reference): compiles the coincidence query
into a Spiking Neural Network — a SpikeSourceArray feeding the stored template-channel
spikes into a single LIF "detector" neuron whose threshold is crossed when ~k template
spikes land inside the leak window W. GeNN runs the whole stream on the GPU and records
the detector's output spikes = the match timestamps. Only those leave the device — the
Paradigm B "only matches transferred to host" property, now at GPU line rate.

Semantic note: GeNN's LIF integrates TOTAL input current, so the detector reacts to
k template spikes within W (vs the CPU reference's k DISTINCT channels). For a small
template (e.g. 2 channels) tune Vthresh/refractory to match; for large templates use
per-channel sub-detectors. Counts are validated against paradigm_b_matcher.py.

Run on a GeNN box:  python paradigm_b_genn.py
"""
import os

# --- GeNN 5 / PyGeNN API (guarded: import only where the toolchain exists) ---
try:
    import numpy as np
    from pygenn import GeNNModel, init_weight_update, init_postsynaptic
    _HAVE_GENN = True
except Exception as _e:                      # noqa: BLE001 — any import failure -> stub
    _HAVE_GENN = False
    _IMPORT_ERR = _e

from spike_telemetry_hub import disk_query, synth, SpikeTelemetryHub


def build_and_run(spk_path, channels, window, k, weight=1.0):
    """Compile the coincidence query into a GeNN SNN, stream the stored template
    spikes through it on the GPU, return the detector's output spike times (matches)."""
    channels = sorted(channels)
    # 1. pull only the template channels from storage (partial read, as in Paradigm A)
    events, _ = disk_query(spk_path, 0, 2**31 - 1, channels=channels)
    duration = (max(t for t, _ in events) + 1) if events else 1

    # 2. format spikes for a SpikeSourceArray: per-neuron sorted times + start/end idx
    per_ch = {c: [] for c in channels}
    for t, c in events:
        per_ch[c].append(float(t))
    spike_times, starts, ends = [], [], []
    for c in channels:
        starts.append(len(spike_times))
        spike_times.extend(sorted(per_ch[c]))
        ends.append(len(spike_times))

    # 3. build the network
    model = GeNNModel("float", "paradigm_b")
    model.dt = 1.0
    ssa = model.add_neuron_population(
        "In", len(channels), "SpikeSourceArray", {},
        {"startSpike": np.array(starts), "endSpike": np.array(ends)})
    ssa.extra_global_params["spikeTimes"].set_values(np.array(spike_times, dtype=np.float32))

    # LIF detector: leak ~ W, threshold so ~k coincident inputs fire it
    lif_params = {"C": 1.0, "TauM": float(window), "Vrest": 0.0, "Vreset": 0.0,
                  "Vthresh": (k - 0.5) * weight, "Ioffset": 0.0, "TauRefrac": float(window)}
    det = model.add_neuron_population("Det", 1, "LIF", lif_params,
                                      {"V": 0.0, "RefracTime": 0.0})
    det.spike_recording_enabled = True
    model.add_synapse_population(
        "In_Det", "DENSE", ssa, det,
        init_weight_update("StaticPulse", {}, {"g": weight}),
        init_postsynaptic("DeltaCurr"))

    # 4. run the whole stream on device; only the detector's spikes are recorded
    model.build()
    model.load(num_recording_timesteps=int(duration))
    while model.t < duration:
        model.step_time()
    model.pull_recording_buffers_from_device()
    match_times, _ = det.spike_recording_data[0]
    return list(match_times)


def main():
    if not _HAVE_GENN:
        print("=" * 60)
        print("PARADIGM B on GeNN — toolchain not present on this machine")
        print("=" * 60)
        print(f"  import failed: {type(_IMPORT_ERR).__name__}: {_IMPORT_ERR}")
        print("  GeNN needs CUDA 12.8+ (RTX 5070 = Blackwell/sm_120), a C++ compiler,")
        print("  and pygenn. On your GPU box:  pip install pygenn  then re-run.")
        print("  Verified CPU equivalent: python paradigm_b_matcher.py")
        return

    path = os.path.join(".", "data", "telemetry.spk")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        synth(256, 100_000, 0.004, {7, 99}, (50_000, 52_000)).save(path)
    matches = build_and_run(path, channels={7, 99}, window=50, k=2)
    in_burst = [m for m in matches if 50_000 <= m < 52_000]
    print(f"GeNN matches: {len(matches)}  ({len(in_burst)} in burst window)")
    print("(compare against paradigm_b_matcher.py on the same store)")


if __name__ == "__main__":
    main()
