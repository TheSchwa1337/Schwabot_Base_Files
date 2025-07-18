import traceback

import matplotlib.pyplot as plt
import numpy as np

from core.advanced_drift_shell_integration import (  # !/usr/bin/env python3; Add parent directory to path for imports
    Comprehensive,
    Demo,
    Differential,
    Engine,
    Implementation,
    Pattern,
    Shift,
    States,
    This,
    """,
    -,
    __file__,
    all,
    analysis.,
    and,
    application,
    demo,
    differential,
    dynamics,
    error,
    from,
    import,
    in,
    info,
    os,
    os.path.abspath,
    os.path.dirname,
    practical,
    safe_print,
    showcases,
    six,
    states,
    success,
    sys,
    sys.path.append,
    their,
    trading,
    utils.safe_print,
)

    ShiftPatternEngine,
)


class ShiftPatternDemo:
    """Comprehensive demo of all differential states in the Shift Pattern Engine."""

    def __init__(self):
        """Initialize the demo with realistic trading data."""
        self.engine = ShiftPatternEngine(
            shift_durations={
                "BTC": {"short": 16, "mid": 72, "long": 672},
                "XRP": {"short": 12, "mid": 48, "long": 480},
                "ETH": {"short": 14, "mid": 60, "long": 600},
            },
            decay_rate=0.1,
            coherence_threshold=0.05,
        )

        # Generate realistic trading data
        self.generate_trading_data()

    def generate_trading_data(self):
        """Generate realistic trading data for demonstration."""
        np.random.seed(42)  # For reproducible results

        # Generate 1000 time steps
        self.time_steps = 1000
        self.tick_counts = np.arange(self.time_steps)

        # Generate realistic price movements with trends and cycles
        self.prices = 100 + np.cumsum(np.random.normal(0, 0.5, self.time_steps))

        # Add cyclical patterns (Ferris Wheel effect)
        cycle_period = 144
        self.prices += 10 * np.sin(2 * np.pi * self.tick_counts / cycle_period)

        # Generate volume data with correlation to price movements
        self.volumes = np.abs(np.random.normal(1000, 200, self.time_steps))
        self.volumes += 500 * np.abs(np.diff(self.prices, prepend=self.prices[0]))

        # Generate volatility data
        self.volatilities = np.abs(np.random.normal(0.02, 0.01, self.time_steps))

        # Generate coherence data (decreasing over time with recovery)
        self.coherences = 0.9 * np.exp(-self.tick_counts / 200) + 0.1
        self.coherences += 0.1 * np.sin(2 * np.pi * self.tick_counts / 50)

        # Generate error counts for API penalty demo
        self.error_counts = np.random.poisson(0.1, self.time_steps)

    def demo_ferris_wheel_phases(self):
        """Demonstrate Ferris Wheel phase transitions."""
        info("🎡 Demonstrating Ferris Wheel Phase Transitions")

        phases = []
        shift_types = []

        for tick_count in self.tick_counts:
            phase = self.engine.compute_ferris_wheel_phase(tick_count)
            phases.append(phase)

            if tick_count > 0:
                shift_type = self.engine.detect_phase_shift(phase, phases[-2])
                shift_types.append(shift_type)
            else:
                shift_types.append("initial")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Phase plot
        ax1.plot(self.tick_counts, phases, "b-", linewidth=2, label="Phase")
        ax1.set_ylabel("Phase (radians)")
        ax1.set_title("Ferris Wheel Phase Transitions")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Shift types
        colors = {
            "ascent_peak": "green",
            "peak_descent": "red",
            "descent_trough": "orange",
            "trough_ascent": "blue",
            "initial": "gray",
        }
        for i, shift_type in enumerate(shift_types):
            if shift_type in colors:
                ax2.scatter(i, 0, c=colors[shift_type], s=50, alpha=0.7)

        ax2.set_ylabel("Shift Type")
        ax2.set_xlabel("Time Step")
        ax2.set_title("Phase Shift Types")
        ax2.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig("ferris_wheel_phases.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print statistics
        shift_counts = {}
        for shift_type in shift_types:
            shift_counts[shift_type] = shift_counts.get(shift_type, 0) + 1

        safe_print("Phase shift statistics:")
        for shift_type, count in shift_counts.items():
            safe_print(f"  {shift_type}: {count} occurrences")

    def demo_tensor_decay_patterns(self):
        """Demonstrate recursive tensor decay patterns."""
        info("🧠 Demonstrating Recursive Tensor Decay Patterns")

        decay_weights = []
        feedback_values = []

        for i in range(20):  # Show 20 time steps
            weight = self.engine.compute_tensor_decay_weight(i)
            decay_weights.append(weight)

            # Simulate feedback computation
            feedback = weight * np.random.normal(0, 1)  # Simplified feedback
            feedback_values.append(feedback)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Decay weights
        ax1.plot(range(20), decay_weights, "r-", linewidth=2, marker="o")
        ax1.set_xlabel("Time Index (i)")
        ax1.set_ylabel("Decay Weight (w_i)")
        ax1.set_title("Tensor Decay Weights: w_i = e^(-i * λ)")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Feedback values
        ax2.plot(range(20), feedback_values, "g-", linewidth=2, marker="s")
        ax2.set_xlabel("Time Index (i)")
        ax2.set_ylabel("Feedback Value")
        ax2.set_title("Recursive Feedback Values")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("tensor_decay_patterns.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print mathematical details
        safe_print(f"Decay rate (λ): {self.engine.decay_rate}")
        safe_print(f"Initial weight (w_0): {decay_weights[0]:.4f}")
        safe_print(f"Final weight (w_19): {decay_weights[-1]:.4f}")
        safe_print(
            f"Weight ratio (w_19/w_0): {decay_weights[-1] / decay_weights[0]:.4f}"
        )

    def demo_thermal_shift_logic(self):
        """Demonstrate thermal shift logic."""
        info("🌡️ Demonstrating Thermal Shift Logic")

        thermal_pressures = []
        volume_emas = []

        # Compute EMA of volumes
        ema_alpha = 0.1
        ema = self.volumes[0]

        for i, volume in enumerate(self.volumes):
            ema = ema_alpha * volume + (1 - ema_alpha) * ema
            volume_emas.append(ema)

            pressure = self.engine.compute_thermal_pressure(
                volume_ema=ema, volatility=self.volatilities[i]
            )
            thermal_pressures.append(pressure)

        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Volume and EMA
        ax1.plot(self.tick_counts, self.volumes, "b-", alpha=0.5, label="Volume")
        ax1.plot(self.tick_counts, volume_emas, "r-", linewidth=2, label="Volume EMA")
        ax1.set_ylabel("Volume")
        ax1.set_title("Volume and EMA")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volatility
        ax2.plot(self.tick_counts, self.volatilities, "orange", linewidth=2)
        ax2.set_ylabel("Volatility")
        ax2.set_title("Volatility")
        ax2.grid(True, alpha=0.3)

        # Thermal pressure
        ax3.plot(self.tick_counts, thermal_pressures, "purple", linewidth=2)
        ax3.set_ylabel("Thermal Pressure")
        ax3.set_xlabel("Time Step")
        ax3.set_title("Thermal Pressure: P = tanh(V/EMA_V + ε) * (1 + log(1 + σ))")
        ax3.grid(True, alpha=0.3)

        # Pressure vs Volume scatter
        ax4.scatter(
            volume_emas,
            thermal_pressures,
            c=self.volatilities,
            cmap="viridis",
            alpha=0.6,
        )
        ax4.set_xlabel("Volume EMA")
        ax4.set_ylabel("Thermal Pressure")
        ax4.set_title("Pressure vs Volume (colored by volatility)")
        plt.colorbar(ax4.collections[0], ax=ax4, label="Volatility")

        plt.tight_layout()
        plt.savefig("thermal_shift_logic.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print statistics
        safe_print(f"Average thermal pressure: {np.mean(thermal_pressures):.4f}")
        safe_print(
            f"Pressure range: [{np.min(thermal_pressures):.4f}, {np.max(thermal_pressures):.4f}]"
        )
        safe_print(f"Pressure std dev: {np.std(thermal_pressures):.4f}")

    def demo_entropy_coherence_shifts(self):
        """Demonstrate entropy-coherence shift zones."""
        info("🌀 Demonstrating Entropy-Coherence Shift Zones")

        shift_triggers = []
        coherence_deltas = []

        for i in range(1, len(self.coherences)):
            current_coherence = self.coherences[i]
            previous_coherence = self.coherences[i - 1]

            should_trigger = self.engine.compute_entropy_coherence_shift(
                current_coherence, previous_coherence
            )
            shift_triggers.append(should_trigger)

            coherence_delta = current_coherence - previous_coherence
            coherence_deltas.append(coherence_delta)

        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Coherence over time
        ax1.plot(self.tick_counts, self.coherences, "b-", linewidth=2)
        ax1.set_ylabel("Coherence")
        ax1.set_title("Coherence Evolution Over Time")
        ax1.grid(True, alpha=0.3)

        # Coherence delta
        ax2.plot(self.tick_counts[1:], coherence_deltas, "orange", linewidth=2)
        ax2.axhline(
            y=-self.engine.coherence_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold (-{self.engine.coherence_threshold})",
        )
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.set_ylabel("Coherence Delta")
        ax2.set_title("Coherence Delta: C_new - C_prev")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Shift triggers
        trigger_indices = [i for i, trigger in enumerate(shift_triggers) if trigger]
        ax3.scatter(
            trigger_indices, [1] * len(trigger_indices), c="red", s=100, alpha=0.7
        )
        ax3.set_ylabel("Shift Trigger")
        ax3.set_xlabel("Time Step")
        ax3.set_title("Entropy-Coherence Shift Triggers")
        ax3.set_ylim(0.5, 1.5)
        ax3.set_yticks([1])
        ax3.set_yticklabels(["Triggered"])

        plt.tight_layout()
        plt.savefig("entropy_coherence_shifts.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print statistics
        trigger_count = sum(shift_triggers)
        safe_print(f"Total shift triggers: {trigger_count}")
        safe_print(f"Trigger rate: {trigger_count / len(shift_triggers) * 100:.2f}%")
        safe_print(f"Coherence threshold: {self.engine.coherence_threshold}")

    def demo_api_penalty_decay(self):
        """Demonstrate API reflection penalty decay."""
        info("🧠 Demonstrating API Reflection Penalty Decay")

        confidences = []
        penalized_confidences = []

        initial_confidence = 0.9
        current_confidence = initial_confidence

        for error_count in self.error_counts:
            penalized = self.engine.compute_api_penalty_decay(
                confidence=current_confidence, error_count=error_count
            )

            confidences.append(current_confidence)
            penalized_confidences.append(penalized)

            # Update confidence for next iteration
            current_confidence = penalized

        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Error counts
        ax1.plot(
            self.tick_counts,
            self.error_counts,
            "red",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax1.set_ylabel("Error Count")
        ax1.set_title("API Error Counts Over Time")
        ax1.grid(True, alpha=0.3)

        # Confidence evolution
        ax2.plot(
            self.tick_counts,
            confidences,
            "blue",
            linewidth=2,
            label="Current Confidence",
        )
        ax2.plot(
            self.tick_counts,
            penalized_confidences,
            "orange",
            linewidth=2,
            label="Penalized Confidence",
        )
        ax2.set_ylabel("Confidence")
        ax2.set_title("Confidence Evolution: C * e^(-N_errors / τ)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Confidence vs Error scatter
        ax3.scatter(
            self.error_counts,
            penalized_confidences,
            c=self.tick_counts,
            cmap="viridis",
            alpha=0.6,
        )
        ax3.set_xlabel("Error Count")
        ax3.set_ylabel("Penalized Confidence")
        ax3.set_title("Confidence vs Error Count (colored by time)")
        plt.colorbar(ax3.collections[0], ax=ax3, label="Time Step")

        plt.tight_layout()
        plt.savefig("api_penalty_decay.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print statistics
        safe_print(f"Initial confidence: {initial_confidence:.4f}")
        safe_print(f"Final confidence: {penalized_confidences[-1]:.4f}")
        safe_print(
            f"Total confidence loss: {initial_confidence - penalized_confidences[-1]:.4f}"
        )
        safe_print(f"Average error count: {np.mean(self.error_counts):.2f}")

    def demo_time_lock_phase_drift(self):
        """Demonstrate recursive time lock phase drift."""
        info("🎛️ Demonstrating Recursive Time Lock Phase Drift")

        drift_magnitudes = []
        drift_directions = []
        synchronization_triggers = []

        for i in range(len(self.tick_counts)):
            # Generate phases for different time scales
            short_phase = self.engine.compute_ferris_wheel_phase(i, period=16)
            mid_phase = self.engine.compute_ferris_wheel_phase(i, period=72)
            long_phase = self.engine.compute_ferris_wheel_phase(i, period=672)

            drift_magnitude, drift_direction = (
                self.engine.compute_time_lock_phase_drift(
                    short_phase, mid_phase, long_phase
                )
            )

            drift_magnitudes.append(drift_magnitude)
            drift_directions.append(drift_direction)

            # Check for synchronization trigger
            should_sync = drift_magnitude > 0.5  # Threshold
            synchronization_triggers.append(should_sync)

        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Individual phases
        short_phases = [
            self.engine.compute_ferris_wheel_phase(i, period=16)
            for i in self.tick_counts
        ]
        mid_phases = [
            self.engine.compute_ferris_wheel_phase(i, period=72)
            for i in self.tick_counts
        ]
        long_phases = [
            self.engine.compute_ferris_wheel_phase(i, period=672)
            for i in self.tick_counts
        ]

        ax1.plot(self.tick_counts, short_phases, "red", linewidth=2, label="Short-term")
        ax1.plot(self.tick_counts, mid_phases, "green", linewidth=2, label="Mid-term")
        ax1.plot(self.tick_counts, long_phases, "blue", linewidth=2, label="Long-term")
        ax1.set_ylabel("Phase (radians)")
        ax1.set_title("Multi-scale Phases")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drift magnitude
        ax2.plot(self.tick_counts, drift_magnitudes, "purple", linewidth=2)
        ax2.axhline(
            y=0.5, color="red", linestyle="--", alpha=0.7, label="Sync Threshold"
        )
        ax2.set_ylabel("Drift Magnitude")
        ax2.set_title("Time Lock Phase Drift Magnitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Drift direction
        ax3.plot(
            self.tick_counts,
            drift_directions,
            "orange",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax3.set_ylabel("Drift Direction")
        ax3.set_xlabel("Time Step")
        ax3.set_title("Time Lock Phase Drift Direction")
        ax3.grid(True, alpha=0.3)

        # Synchronization triggers
        trigger_indices = [
            i for i, trigger in enumerate(synchronization_triggers) if trigger
        ]
        ax4.scatter(
            trigger_indices, [1] * len(trigger_indices), c="red", s=100, alpha=0.7
        )
        ax4.set_ylabel("Synchronization")
        ax4.set_xlabel("Time Step")
        ax4.set_title("Synchronization Triggers")
        ax4.set_ylim(0.5, 1.5)
        ax4.set_yticks([1])
        ax4.set_yticklabels(["Triggered"])

        plt.tight_layout()
        plt.savefig("time_lock_phase_drift.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print statistics
        sync_count = sum(synchronization_triggers)
        safe_print(f"Total synchronization triggers: {sync_count}")
        safe_print(
            f"Sync rate: {sync_count / len(synchronization_triggers) * 100:.2f}%"
        )
        safe_print(f"Average drift magnitude: {np.mean(drift_magnitudes):.4f}")
        safe_print(
            f"Drift direction distribution: {np.bincount(np.array(drift_directions) + 1)}"
        )

    def demo_asset_specific_durations(self):
        """Demonstrate asset-specific shift durations."""
        info("📊 Demonstrating Asset-Specific Shift Durations")

        assets = ["BTC", "XRP", "ETH"]
        shift_types = ["short", "mid", "long"]

        # Create duration matrix
        duration_matrix = []
        for asset in assets:
            row = []
            for shift_type in shift_types:
                duration = self.engine.get_shift_duration(asset, shift_type)
                row.append(duration)
            duration_matrix.append(row)

        # Plot results
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        im = ax.imshow(duration_matrix, cmap="viridis", aspect="auto")

        # Add text annotations
        for i in range(len(assets)):
            for j in range(len(shift_types)):
                ax.text(
                    j,
                    i,
                    duration_matrix[i][j],
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

        ax.set_xticks(range(len(shift_types)))
        ax.set_yticks(range(len(assets)))
        ax.set_xticklabels(shift_types)
        ax.set_yticklabels(assets)
        ax.set_xlabel("Shift Type")
        ax.set_ylabel("Asset")
        ax.set_title("Asset-Specific Shift Durations")

        plt.colorbar(im, ax=ax, label="Duration (ticks/hours)")
        plt.tight_layout()
        plt.savefig("asset_specific_durations.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print detailed information
        safe_print("Asset-specific shift durations:")
        for asset in assets:
            safe_print(f"  {asset}:")
            for shift_type in shift_types:
                duration = self.engine.get_shift_duration(asset, shift_type)
                safe_print(f"    {shift_type}: {duration} ticks/hours")

    def run_comprehensive_demo(self):
        """Run all demonstrations."""
        success("🚀 Starting Comprehensive Shift Pattern Engine Demo")

        # Run all demos
        self.demo_ferris_wheel_phases()
        self.demo_tensor_decay_patterns()
        self.demo_thermal_shift_logic()
        self.demo_entropy_coherence_shifts()
        self.demo_api_penalty_decay()
        self.demo_time_lock_phase_drift()
        self.demo_asset_specific_durations()

        success(
            "✅ Comprehensive demo completed! Check the generated PNG files for visualizations."
        )

        # Print summary statistics
        safe_print("\n📈 Demo Summary Statistics:")
        safe_print(f"  Total time steps: {self.time_steps}")
        safe_print(
            f"  Price range: [{np.min(self.prices):.2f}, {np.max(self.prices):.2f}]"
        )
        safe_print(
            f"  Volume range: [{np.min(self.volumes):.0f}, {np.max(self.volumes):.0f}]"
        )
        safe_print(f"  Average volatility: {np.mean(self.volatilities):.4f}")
        safe_print(f"  Average coherence: {np.mean(self.coherences):.4f}")
        safe_print(f"  Total errors: {np.sum(self.error_counts)}")


def main():
    """Main demo function."""
    try:
        demo = ShiftPatternDemo()
        demo.run_comprehensive_demo()
    except Exception as e:
        error(f"Demo failed: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
