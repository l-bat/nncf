import sqlite3
conn = sqlite3.connect("mlflow.db")
c = conn.cursor()
c.execute("SELECT r.run_uuid FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id WHERE e.name = 'fqlora_Llama-3.2-3B-Instruct' AND r.status = 'RUNNING' ORDER BY r.start_time DESC LIMIT 1")
rid = c.fetchone()[0]

# Get loss at key milestones
c.execute("SELECT step, value FROM metrics WHERE run_uuid = ? AND key = 'loss' ORDER BY step", (rid,))
rows = c.fetchall()

print(f"Run: {rid[:12]}  Total steps logged: {len(rows)}")
print(f"\n--- Loss at milestones ---")
milestones = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, len(rows)]
for m in milestones:
    if m <= len(rows):
        step, val = rows[m-1]
        print(f"  step {step:5d}: loss = {val:.4f}")

# Compute average loss in windows
print(f"\n--- Average loss per 100-step window ---")
for start in range(0, len(rows), 100):
    window = rows[start:start+100]
    if window:
        avg = sum(v for _, v in window) / len(window)
        print(f"  steps {window[0][0]:4d}-{window[-1][0]:4d}: avg_loss = {avg:.4f}  (n={len(window)})")

# Last 50 steps trend
last50 = rows[-50:]
first25 = last50[:25]
second25 = last50[25:]
avg1 = sum(v for _, v in first25) / len(first25)
avg2 = sum(v for _, v in second25) / len(second25)
print(f"\n--- Recent trend (last 50 steps) ---")
print(f"  First half avg:  {avg1:.4f}  (steps {first25[0][0]}-{first25[-1][0]})")
print(f"  Second half avg: {avg2:.4f}  (steps {second25[0][0]}-{second25[-1][0]})")
print(f"  Delta: {avg2 - avg1:+.4f}")

# Min loss
min_step, min_val = min(rows, key=lambda x: x[1])
print(f"\n  Best loss: {min_val:.4f} at step {min_step}")
print(f"  Latest:    {rows[-1][1]:.4f} at step {rows[-1][0]}")

# ── Gradient norms (vanishing / exploding diagnosis) ──
c.execute("SELECT DISTINCT key FROM metrics WHERE run_uuid = ? AND key LIKE '%grad_norm%'", (rid,))
grad_keys = [r[0] for r in c.fetchall()]

if not grad_keys:
    print(f"\n--- Gradient norms: no grad_norm metrics found ---")
else:
    import re
    from collections import defaultdict

    # Sample at ~10 evenly spaced steps across training
    all_steps = [s for s, _ in rows]
    sample_steps = sorted(set(
        all_steps[i * (len(all_steps) - 1) // 9] for i in range(10)
    )) if len(all_steps) >= 10 else all_steps

    # Group keys by param type (lora_A_grad_norm, lora_B_grad_norm, scale_grad_norm, etc.)
    # key format: quantizers/<layer_name>/<param>_grad_norm
    param_types = sorted(set(k.split("/")[-1] for k in grad_keys))

    for param_type in param_types:
        keys_for_type = sorted(k for k in grad_keys if k.endswith(param_type))
        if not keys_for_type:
            continue

        # Load latest value for each layer
        layer_latest = {}
        layer_data = {}  # key -> {step: value}
        for key in keys_for_type:
            c.execute("SELECT step, value FROM metrics WHERE run_uuid = ? AND key = ? ORDER BY step", (rid, key))
            layer_rows = c.fetchall()
            if layer_rows:
                layer_data[key] = dict(layer_rows)
                layer_latest[key] = layer_rows[-1][1]

        if not layer_latest:
            continue

        values_latest = list(layer_latest.values())
        mn = min(values_latest)
        mx = max(values_latest)
        mean = sum(values_latest) / len(values_latest)

        # Find outlier layers
        vanishing = [k for k, v in layer_latest.items() if v < 1e-6]
        exploding = [k for k, v in layer_latest.items() if v > 10.0]

        print(f"\n{'='*60}")
        print(f"  {param_type}  ({len(keys_for_type)} layers)")
        print(f"  Latest — min={mn:.2e}  max={mx:.2e}  mean={mean:.2e}  ratio={mx/mn if mn>0 else float('inf'):.1f}x")

        if vanishing:
            print(f"  ⚠ VANISHING layers ({len(vanishing)}):  {', '.join(k.split('/')[1] for k in vanishing[:5])}")
        if exploding:
            print(f"  ⚠ EXPLODING layers ({len(exploding)}):  {', '.join(k.split('/')[1] for k in exploding[:5])}")

        # Show distribution at each sampled step: min / mean / max across all layers
        # print(f"\n  {'step':>6}  {'min':>10}  {'mean':>10}  {'max':>10}  {'outlier layer (max)'}")
        # for step in sample_steps:
        #     step_vals = {}
        #     for key, d in layer_data.items():
        #         closest = min(d.keys(), key=lambda s: abs(s - step))
        #         step_vals[key] = d[closest]
        #     if not step_vals:
        #         continue
        #     sv = list(step_vals.values())
        #     s_mn, s_mx, s_mean = min(sv), max(sv), sum(sv)/len(sv)
        #     worst_layer = max(step_vals, key=step_vals.get).split("/")[1] if step_vals else ""
        #     print(f"  {step:6d}  {s_mn:10.3e}  {s_mean:10.3e}  {s_mx:10.3e}  {worst_layer}")

        # Per-layer summary sorted by latest grad norm (highlights outliers at bottom/top)
        # print(f"\n  Per-layer latest grad norms (sorted):")
        # for key, val in sorted(layer_latest.items(), key=lambda x: x[1]):
        #     layer_name = key.split("/")[1] if "/" in key else key
        #     flag = "  ⚠ VANISHING" if val < 1e-6 else ("  ⚠ EXPLODING" if val > 10.0 else "")
        #     print(f"    {layer_name:<55} {val:.3e}{flag}")

conn.close()
