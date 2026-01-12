import numpy as np
import matplotlib.pyplot as plt

class TrajectoryVisualizer:
    def __init__(self, ground_truth=None):
        self.ground_truth = ground_truth
        self.fig = None
        self.ax = None
        
    def setup_plot(self, title="Ground Truth vs Estimated Trajectory"):
        """Initialize the 2D plot"""
        plt.ion()  # Turn on interactive mode
        
        self.fig = plt.figure(figsize=(7, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Plot ground truth
        self._plot_ground_truth()
        
        plt.show(block=False)
        plt.draw()
        
    def _plot_ground_truth(self):
        """Plot the ground truth trajectory"""
        xs = self.ground_truth[:, 0, 3]
        zs = self.ground_truth[:, 2, 3]
        self.ax.plot(xs, zs, c='dimgray', linewidth=2, 
                    label=f'Ground Truth ({len(self.ground_truth)} frames)')
    
    def update_trajectory(self, trajectory, frame_idx=None):
        """Update the estimated trajectory plot"""
        if frame_idx is None:
            frame_idx = len(trajectory) - 1
            
        xs = trajectory[:frame_idx+1, 0, 3]
        zs = trajectory[:frame_idx+1, 2, 3]
        
        self.ax.plot(xs, zs, c='darkorange', linewidth=2, 
                   label='Estimated Trajectory' if frame_idx == 0 else '')
        if frame_idx == 0:
            self.ax.legend()
        
        plt.pause(1e-32)
    
    def show_final_plot(self):
        """Show the final plot"""
        self.ax.legend()
        plt.show()
    
    def save_plot(self, filename):
        """Save the plot to file"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")