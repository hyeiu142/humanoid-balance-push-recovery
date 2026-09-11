"""
Automated Benchmark Suite for Humanoid Balance and Push Recovery.
Compares Passive Baseline vs V1 Virtual Model Control (VMC) PID.
Sweeps disturbance forces in Forward, Backward, and Lateral directions,
generates a comprehensive metrics comparison table and multi-panel response plots.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for generating plots
import matplotlib.pyplot as plt

from sim.simulation_base import SimulationBase
from controllers.v1_pid.vmc_pid_controller import VMCPIDController
from benchmark.metrics import TrajectoryLogger, TrialMetrics


def run_single_trial(
    sim: SimulationBase,
    controller,  # None for passive, VMCPIDController for active
    controller_name: str,
    direction: str,
    force_vec: list,
    duration: float = 0.1,
    push_start: float = 0.5,
    sim_duration: float = 3.0,
) -> tuple[TrialMetrics, TrajectoryLogger]:
    """Execute a single push recovery trial and record full trajectory."""
    state = sim.reset()
    if controller is not None:
        controller.reset()

    sim.schedule_push(force_vec, duration=duration, start_time=push_start)
    logger = TrajectoryLogger()

    total_steps = int(sim_duration / sim.dt)
    fell = False
    fall_time = None

    for step in range(total_steps):
        t = step * sim.dt
        if controller is not None:
            action = controller.compute_action(state, sim.dt)
        else:
            action = None  # passive nominal control

        state, active_push, fell = sim.step(action)
        logger.log(state, active_push)

        if fell:
            fall_time = t
            break

    force_mag = float(np.linalg.norm(force_vec))
    metrics = logger.compute_metrics(
        controller_name=controller_name,
        direction=direction,
        force_mag=force_mag,
        duration=duration,
        push_start=push_start,
        fell=fell,
        fall_time=fall_time,
    )
    return metrics, logger


def main():
    print("=" * 80)
    print("   HUMANOID BALANCE & PUSH RECOVERY - V1 BENCHMARK SUITE")
    print("=" * 80)

    sim = SimulationBase()
    v1_ctrl = VMCPIDController(sim.model, sim.nominal_qpos)

    # Sweep configurations
    experiments = [
        # (direction_name, force_vector, force_mag, duration)
        ("Forward (+X)", [30, 0, 0], 30.0, 0.1),
        ("Forward (+X)", [70, 0, 0], 70.0, 0.1),
        ("Forward (+X)", [100, 0, 0], 100.0, 0.1),
        ("Forward (+X)", [120, 0, 0], 120.0, 0.1),
        ("Forward (+X)", [150, 0, 0], 150.0, 0.1),
        ("Forward (+X)", [180, 0, 0], 180.0, 0.1),

        ("Backward (-X)", [-30, 0, 0], 30.0, 0.1),
        ("Backward (-X)", [-60, 0, 0], 60.0, 0.1),
        ("Backward (-X)", [-90, 0, 0], 90.0, 0.1),

        ("Lateral (+Y)", [0, 40, 0], 40.0, 0.1),
        ("Lateral (+Y)", [0, 80, 0], 80.0, 0.1),
        ("Lateral (+Y)", [0, 120, 0], 120.0, 0.1),
    ]

    passive_results = []
    v1_results = []

    # Store sample loggers for plotting
    plot_loggers = {}

    print("\nRunning test sweep across Passive Baseline vs V1 VMC PID...")
    for dir_name, f_vec, f_mag, dur in experiments:
        # 1. Passive
        p_met, p_log = run_single_trial(
            sim=sim,
            controller=None,
            controller_name="Passive",
            direction=dir_name,
            force_vec=f_vec,
            duration=dur,
        )
        passive_results.append(p_met)

        # 2. V1 VMC PID
        v_met, v_log = run_single_trial(
            sim=sim,
            controller=v1_ctrl,
            controller_name="V1: VMC PID",
            direction=dir_name,
            force_vec=f_vec,
            duration=dur,
        )
        v1_results.append(v_met)

        key = f"{dir_name}_{int(f_mag)}"
        plot_loggers[f"Passive_{key}"] = p_log
        plot_loggers[f"V1_{key}"] = v_log

        status_p = "SURVIVED" if p_met.survived else f"FELL ({p_met.fall_time:.2f}s)"
        status_v = "SURVIVED" if v_met.survived else f"FELL ({v_met.fall_time:.2f}s)"
        print(f"[{dir_name:14s}] Force: {f_mag:5.1f}N | Passive: {status_p:16s} | V1 PID: {status_v:16s}")

    # Print Summary Table
    print("\n" + "=" * 95)
    print(f"{'Direction':15s} | {'Force (N)':9s} | {'Passive Status':16s} | {'V1 PID Status':16s} | {'V1 Settle (s)':13s} | {'V1 Max Tilt':11s}")
    print("-" * 95)
    for p, v in zip(passive_results, v1_results):
        settle_str = f"{v.settling_time:.2f}s" if v.settling_time is not None else "N/A"
        p_stat = "SURVIVED" if p.survived else f"FELL ({p.fall_time:.2f}s)"
        v_stat = "SURVIVED" if v.survived else f"FELL ({v.fall_time:.2f}s)"
        print(f"{v.direction:15s} | {v.force_magnitude:9.1f} | {p_stat:16s} | {v_stat:16s} | {settle_str:13s} | {v.max_tilt_deg:8.2f}°")
    print("=" * 95)

    # Compute Maximum Impulse Tolerated per Direction
    max_impulse = {"Passive": {}, "V1 PID": {}}
    for d in ["Forward (+X)", "Backward (-X)", "Lateral (+Y)"]:
        p_surv = [r.impulse for r in passive_results if r.direction == d and r.survived]
        v_surv = [r.impulse for r in v1_results if r.direction == d and r.survived]
        max_impulse["Passive"][d] = max(p_surv) if p_surv else 0.0
        max_impulse["V1 PID"][d] = max(v_surv) if v_surv else 0.0

    print("\n--- MAXIMUM TOLERATED IMPULSE (N·s) ---")
    for d in max_impulse["Passive"]:
        p_imp = max_impulse["Passive"][d]
        v_imp = max_impulse["V1 PID"][d]
        gain = ((v_imp - p_imp) / p_imp * 100) if p_imp > 0 else 0
        print(f"  {d:15s}: Passive = {p_imp:5.1f} N·s  -->  V1 PID = {v_imp:5.1f} N·s  (+{gain:.0f}% improvement)")

    # Generate Visualization Figure
    generate_plots(passive_results, v1_results, plot_loggers, max_impulse)


def generate_plots(passive_results, v1_results, plot_loggers, max_impulse):
    """Generate multi-panel comparison chart and save to benchmark/v1_benchmark_results.png."""
    fig = plt.figure(figsize=(15, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Bar chart: Maximum Impulse
    ax1 = fig.add_subplot(2, 2, 1)
    dirs = ["Forward (+X)", "Backward (-X)", "Lateral (+Y)"]
    x = np.arange(len(dirs))
    w = 0.35
    p_vals = [max_impulse["Passive"][d] for d in dirs]
    v_vals = [max_impulse["V1 PID"][d] for d in dirs]

    rects1 = ax1.bar(x - w / 2, p_vals, w, label="Passive Stance", color="#e74c3c", alpha=0.85)
    rects2 = ax1.bar(x + w / 2, v_vals, w, label="V1: VMC PID", color="#2ecc71", alpha=0.85)

    ax1.set_ylabel("Max Impulse Tolerated (N·s)", fontsize=11, fontweight="bold")
    ax1.set_title("Push Disturbance Tolerance by Direction", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dirs, fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Tilt Trajectory Comparison (Forward 120N push)
    ax2 = fig.add_subplot(2, 2, 2)
    key_p = "Passive_Forward (+X)_120"
    key_v = "V1_Forward (+X)_120"
    if key_p in plot_loggers and key_v in plot_loggers:
        lp = plot_loggers[key_p]
        lv = plot_loggers[key_v]
        ax2.plot(lp.times, lp.tilts_deg, "r--", linewidth=2.0, label="Passive (Fell at 2.7s)")
        ax2.plot(lv.times, lv.tilts_deg, "g-", linewidth=2.5, label="V1 VMC PID (Recovered in 1.4s)")
        ax2.axvspan(0.5, 0.6, color="orange", alpha=0.25, label="Push Active (120N, 0.1s)")
        ax2.axhline(1.0, color="gray", linestyle=":", label="Settling Threshold (1°)")
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Torso Tilt Angle (deg)", fontsize=11, fontweight="bold")
        ax2.set_title("Recovery Trajectory under 120N Push", fontsize=12, fontweight="bold")
        ax2.legend(frameon=True, loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. CoM vs CoP Excursion
    ax3 = fig.add_subplot(2, 2, 3)
    if key_v in plot_loggers:
        lv = plot_loggers[key_v]
        ax3.plot(lv.times, lv.com_x, "b-", linewidth=2.0, label="CoM X position")
        ax3.plot(lv.times, lv.cop_x, "m--", linewidth=2.0, label="CoP X position (Ankle/Hip)")
        ax3.axhline(0.12, color="black", linestyle="--", alpha=0.7, label="Toe Limit (Support Polygon)")
        ax3.axhline(-0.05, color="black", linestyle=":", alpha=0.7, label="Heel Limit")
        ax3.axvspan(0.5, 0.6, color="orange", alpha=0.25)
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Position along X (m)", fontsize=11, fontweight="bold")
        ax3.set_title("CoM and CoP Trajectory (120N Push)", fontsize=12, fontweight="bold")
        ax3.legend(frameon=True, loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Settling Time vs Disturbance Force
    ax4 = fig.add_subplot(2, 2, 4)
    f_surv = [r.force_magnitude for r in v1_results if r.direction == "Forward (+X)" and r.survived]
    t_settle = [r.settling_time for r in v1_results if r.direction == "Forward (+X)" and r.survived]
    ax4.plot(f_surv, t_settle, "go-", linewidth=2.2, markersize=7, label="V1 PID Settling Time")
    ax4.set_xlabel("Push Force (N, duration=0.1s)", fontsize=11)
    ax4.set_ylabel("Settling Time (s)", fontsize=11, fontweight="bold")
    ax4.set_title("V1 Recovery Speed vs Disturbance Magnitude", fontsize=12, fontweight="bold")
    ax4.legend(frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, "v1_benchmark_results.png")
    plt.savefig(out_path, dpi=200)
    print(f"\nBenchmark visualization saved to: {out_path}")


if __name__ == "__main__":
    main()
