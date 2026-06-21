import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union

# Define a unified data structure for passing data to the plotter
@dataclass
class PlotState:
    """Holds the state required for plotting calibration progress."""
    iteration: int
    params: np.ndarray
    param_names: List[str]
    nse: float
    loss: float
    
    # Histories
    iteration_history: List[int]
    nse_history: List[float]
    loss_history: List[float]
    
    # Model Outputs
    obs_valid: np.ndarray
    pred_valid: np.ndarray
    
    # Detailed components (can be None if not available)
    # Using specific fields for common components to avoid dict lookups for everything
    E_rain: Optional[np.ndarray] = None
    E_melt: Optional[np.ndarray] = None
    A_channel: Optional[np.ndarray] = None
    R_rain: Optional[np.ndarray] = None
    R_melt: Optional[np.ndarray] = None
    
    # Extra diagnostics
    diagnostics: Optional[Dict[str, float]] = None
    param_sds: Optional[np.ndarray] = None
    
    model_type: str = 'crsem'

class CalibrationVisualizer:
    """
    Handles real-time visualization of the calibration process.
    """
    
    UNITS_RAINFALL_EROSIVITY = fr'$MJ \cdot mm \cdot ha^{{-1}} \cdot h^{{-1}} \cdot yr^{{-1}}$'

    def __init__(self, model_type: str = 'crsem'):
        self.model_type = model_type.lower()
        self.fig = None
        self.axes = {}

    def initialize(self):
        """Initialize the plot window."""
        plt.ion()
        # Use constrained_layout for better automatic layout handling, especially with twin axes
        self.fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 3)
        
        self.axes['ts'] = self.fig.add_subplot(gs[0, 0])
        self.axes['params'] = self.fig.add_subplot(gs[0, 1])
        self.axes['rainR'] = self.fig.add_subplot(gs[0, 2])
        self.axes['nse'] = self.fig.add_subplot(gs[1, 0])
        # Create twin axis for loss here, once.
        self.axes['loss'] = self.axes['nse'].twinx()
        
        self.axes['month'] = self.fig.add_subplot(gs[1, 1])
        self.axes['annual'] = self.fig.add_subplot(gs[1, 2])
        
        # plt.tight_layout() # constrained_layout handles this better

    def update(self, state: PlotState):
        """Update the plots with new state data."""
        if self.fig is None:
            self.initialize()
            
        # Clear axes
        for ax in self.axes.values():
            if ax is not None:
                ax.clear()
            
        # 1. Time Series
        self._plot_time_series(
            self.axes['ts'], 
            state.obs_valid, 
            state.pred_valid, 
            state.iteration, 
            state.nse
        )
        
        # 2. Parameters
        self._plot_parameters(
            self.axes['params'],
            state.params,
            state.param_names,
            state.diagnostics,
            state.param_sds
        )
        
        # 3. Erosivity (Rain/Melt)
        if state.R_rain is not None:
             self._plot_rain_erosivity_monthly(
                 self.axes['rainR'],
                 state.R_rain,
                 state.R_melt
             )
             
        # 4. Convergence (NSE/Loss)
        self._plot_nse_convergence(
            self.axes['nse'],
            self.axes['loss'],
            state.iteration_history,
            state.nse_history,
            state.loss_history
        )
        
        # 5. Monthly Erosion Components
        if state.E_rain is not None and state.A_channel is not None:
            self._plot_monthly_erosion(
                self.axes['month'],
                state.E_rain,
                state.E_melt,
                state.A_channel
            )
            
        # 6. Annual Totals
        self._plot_annual_totals(
            self.axes['annual'],
            state.obs_valid,
            state.pred_valid
        )
        
        # Refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finalize(self, block: bool = False):
        """Finalize plotting."""
        plt.ioff()
        try:
            plt.show(block=block)
            if not block:
                plt.pause(0.1)
        except Exception as e:
            print(f"[Warning] Failed to finalize matplotlib window: {e}")

    # --- Internal Plotting Methods ---

    def _plot_time_series(self, ax, obs, pred, iter_num, nse):
        time_axis = np.arange(len(obs))
        ax.plot(time_axis, obs, 'b-', label='Observed SSF', alpha=0.7, linewidth=1.5)
        ax.plot(time_axis, pred, 'r--', label='Simulated SSF', linewidth=2)
        
        ax.set_title(f'Iteration {iter_num} | Current NSE: {nse:.4f}')
        ax.set_ylabel('Total SSF (tons)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_parameters(self, ax, params, param_names, diagnostics, param_sds):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title('Current Parameter Values')

        num_params = len(param_names)
        mid_point = (num_params + 1) // 2
        
        def format_params(start_idx, end_idx):
            text = ""
            for i in range(start_idx, end_idx):
                if i >= len(params): break
                name = param_names[i]
                val = params[i]
                if param_sds is not None and i < len(param_sds) and np.isfinite(param_sds[i]):
                    text += f"{name:<9s}: {val:.4f} ± {param_sds[i]:.3g}\n"
                else:
                    text += f"{name:<9s}: {val:.4f}\n"
            return text

        col1_text = format_params(0, mid_point)
        col2_text = format_params(mid_point, num_params)

        ax.text(0.05, 0.92, col1_text, family='monospace', va='top', fontsize=9)
        ax.text(0.55, 0.92, col2_text, family='monospace', va='top', fontsize=9)

        if diagnostics:
            diag_lines = ["Annual Means"]
            for k, v in diagnostics.items():
                if isinstance(v, float):
                    diag_lines.append(f"{k:<10}: {v:.3f}")
                else:
                    diag_lines.append(f"{k:<10}: {v}")
            
            stats_text = "\n".join(diag_lines)
            # Move to bottom to avoid overlap with parameter list
            ax.text(0.05, 0.00, stats_text, family='monospace', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.7))

    def _plot_rain_erosivity_monthly(self, ax, R_rain, R_melt):
        if R_rain is None: return
        
        # Handle cases where R_melt is None (e.g., RUSLE)
        if R_melt is None:
            R_melt = np.zeros_like(R_rain)

        try:
            num_months = len(R_rain)
            if num_months == 0: return
            num_years = max(1, num_months // 12)
            # Truncate to full years
            limit = num_years * 12
            
            R_rain_mean = np.nanmean(R_rain[:limit].reshape(num_years, 12), axis=0)
            R_melt_mean = np.nanmean(R_melt[:limit].reshape(num_years, 12), axis=0)
            
            R_total = float(np.nansum(R_rain_mean) + np.nansum(R_melt_mean))
        except Exception:
             # Fallback
            R_rain_mean = np.zeros(12)
            R_melt_mean = np.zeros(12)
            R_total = 0.0

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.bar(months, R_rain_mean, label='Rain', color='steelblue')
        ax.bar(months, R_melt_mean, bottom=R_rain_mean, label='Snowmelt', color='coral')
        
        ax.set_title('Mean Monthly Erosivity')
        ax.set_ylabel(self.UNITS_RAINFALL_EROSIVITY)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.text(0.98, 0.95, f"Total: {R_total:.1f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.7))

    def _plot_nse_convergence(self, ax, ax_loss, iter_hist, nse_hist, loss_hist):
        if not iter_hist: return

        # NSE on left axis
        ax.plot(iter_hist, nse_hist, 'm-o', label='NSE', markersize=3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.set_ylabel('Nash-Sutcliffe Efficiency', color='m')
        ax.tick_params(axis='y', labelcolor='m')
        
        finite_nse = [v for v in nse_hist if np.isfinite(v)]
        if finite_nse:
             ax.set_ylim(max(min(finite_nse) - 0.1, -1.0), 1.0) # Cap lower bound

        # Loss on right axis
        if loss_hist and ax_loss is not None:
            ax_loss.plot(iter_hist, loss_hist, 'c-o', label='Loss', markersize=3, alpha=0.7)
            ax_loss.set_ylabel('Loss', color='c')
            ax_loss.tick_params(axis='y', labelcolor='c')
            
            # Combine legends (tricky with twinx, just adding another legend often overlaps)
            # Simple approach: let them be separate or just label axes well.
        
        ax.set_title('Optimization Convergence')
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

    def _plot_monthly_erosion(self, ax, E_rain, E_melt, A_channel):
        if E_melt is None: E_melt = np.zeros_like(E_rain)
        
        num_months = len(E_rain)
        if num_months == 0: return
        num_years = max(1, num_months // 12)
        limit = num_years * 12
        
        def get_monthly_mean(arr):
            return np.mean(arr[:limit].reshape(num_years, 12), axis=0)

        E_rain_mean = get_monthly_mean(E_rain)
        E_melt_mean = get_monthly_mean(E_melt)
        A_channel_mean = get_monthly_mean(A_channel)

        A_channel_pos = np.maximum(0, A_channel_mean)
        A_channel_neg = np.minimum(0, A_channel_mean)

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x = np.arange(len(months))
        width = 0.8
        
        ax.bar(x, E_rain_mean, width, label='Rain', color='forestgreen')
        ax.bar(x, E_melt_mean, width, bottom=E_rain_mean, label='Snowmelt', color='coral')
        ax.bar(x, A_channel_pos, width, bottom=E_rain_mean + E_melt_mean, label='Channel Erosion', color='skyblue')
        ax.bar(x, A_channel_neg, width, label='Channel Deposition', color='saddlebrown')

        # Calculate ratios for title
        total_gross = np.sum(E_rain) + np.sum(E_melt)
        if total_gross > 0:
            chan_ratio = (np.sum(A_channel) / total_gross) * 100
            rain_ratio = (np.sum(E_rain) / total_gross) * 100
            melt_ratio = (np.sum(E_melt) / total_gross) * 100
        else: 
            chan_ratio = 0
            rain_ratio = 0
            melt_ratio = 0
            
        title = f'Chan: {chan_ratio:.1f}%, R_rain:{rain_ratio:.1f}%, R_melt:{melt_ratio:.1f}%'
        
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Sediment (tons)')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

    def _plot_annual_totals(self, ax, obs, pred):
        try:
            num_months = len(obs)
            if num_months == 0: return
            num_years = max(1, num_months // 12)
            limit = num_years * 12
            
            obs_annual = np.nansum(obs[:limit].reshape(num_years, 12), axis=1)
            pred_annual = np.nansum(pred[:limit].reshape(num_years, 12), axis=1)
            years_idx = np.arange(1, num_years + 1)

            ax.plot(years_idx, obs_annual, label='Observed', color='royalblue')
            ax.plot(years_idx, pred_annual, label='Simulated', color='salmon')
            ax.set_title('Annual SSF Totals')
            ax.set_ylabel('tons')
            ax.set_xlabel('Year Index')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
        except Exception:
            pass

# Keep initialize_plot_window and finalize_plot_window as module level functions for compatibility if needed,
# or for simple calls. But ideally Calibrator should use the class.
def initialize_plot_window():
    plt.ion()

def finalize_plot_window(block=False):
    plt.ioff()
    plt.show(block=block)
