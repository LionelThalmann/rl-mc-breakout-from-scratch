import os
import pickle
import time
from collections import deque
from pathlib import Path
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from game_env import game_env, conf  
from MC_agent import MC_agent, state_encoder


def make_env(layout, brick_rows=4):
    base_env = game_env() 
    # overwrite the attributes with the desired configuration - to be sure
    cfg = conf(layout, brick_rows=brick_rows)
    base_env.__dict__.update(cfg)
    base_env.reset()
    return base_env


def run_training(env, agent, num_episodes):
    
    returns = []  # Log each G per episode
    episode_infos = []  # to store info from the last episode
    for ep in range(num_episodes):
        init_obs = env.reset()   # reset the env
        action = agent.start_episode(init_obs) # start first episode with that env
        
        # Take the steps until done
        done = False
        G = 0.0            # inital G
        steps = 0
        while not done:
            steps += 1
            obs, reward, done, info = env.step(action) # advance the game
            G += reward   # record reward
            action = agent.step(obs, reward, done)  # advance the agent 
            

        info["steps"] = steps
        episode_infos.append(info)  # store info from the last episode
        # end the episode - agent walks through the timesteps backwards and updates Q
        agent.end_episode()
        
        returns.append(G)
        
        agent.eps = max(0.05, agent.eps * 0.9995) # epsilon decay
        
    return episode_infos, returns
        
        
# Compute best trajectroies for each velocity
def rollout_greedy(env, agent, vx0, max_steps=500):
    traj = []
    obs = env.reset()
    env.ball_vx = vx0           # force start direction
    obs = env.make_obs()        # refresh obs
    state = state_encoder(obs)

    for _ in range(max_steps):
        traj.append((obs['ball_x'], obs['ball_y']))
        action = agent.policy(state)      # greedy (eps=0)
        obs, _, done, _ = env.step(action)
        if done:
            break
        state = state_encoder(obs)
    return traj
        


def plot_trajectories(traj_dict, brick_rows):
    out_dir = Path("plots_f");  out_dir.mkdir(exist_ok=True)
    for layout, by_vx in traj_dict.items():
            plt.figure()
            for vx0, traj in by_vx.items():
                xs, ys = zip(*traj)
                plt.plot(xs, ys, label=f"vx0={vx0:+d}")
            plt.gca().invert_yaxis()
            plt.title(f"Greedy ball trajectories — {layout}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"trajectories_{layout}_{brick_rows}.png")
            plt.close()


def plot_grid_performance(results, brick_rows):
    out_dir = Path("plots_f")
    out_dir.mkdir(exist_ok=True)
    
    for layout in {r["layout"] for r in results}:
        subset = [r for r in results if r["layout"] == layout]
        # Sort by performance
        subset = sorted(subset, key=lambda x: x["mean_return"], reverse=True)
        
        labels = [f"ε={r['eps']:.3f}, γ={r['gamma']:.3f}, α={r['alpha']:.3f}"
                 for r in subset]
        scores = [r["mean_return"] for r in subset]
        
        # Horizontal bars work better for many labels
        fig, ax = plt.subplots(figsize=(10, max(8, len(subset) * 0.3)))
        y_pos = np.arange(len(subset))
        
        ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean Return (last-100)")
        ax.set_title(f"Grid Search Performance - {layout}")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"grid_{layout}_{brick_rows}_2.png", dpi=150, bbox_inches='tight')
        plt.close()
        
def plot_grid_performance_heatmap(results, brick_rows):
    out_dir = Path("plots_f")
    out_dir.mkdir(exist_ok=True)
    
    for layout in {r["layout"] for r in results}:
        subset = [r for r in results if r["layout"] == layout]
        
        # Extract unique parameter values
        eps_values = sorted(set(r['eps'] for r in subset))
        gamma_values = sorted(set(r['gamma'] for r in subset))
        alpha_values = sorted(set(r['alpha'] for r in subset))
        
        # If one parameter is fixed, create 2D heatmap
        if len(alpha_values) == 1:
            # Create matrix for heatmap
            scores_matrix = np.zeros((len(gamma_values), len(eps_values)))
            
            for r in subset:
                i = gamma_values.index(r['gamma'])
                j = eps_values.index(r['eps'])
                scores_matrix[i, j] = r['mean_return']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(scores_matrix, 
                       xticklabels=[f"{e:.3f}" for e in eps_values],
                       yticklabels=[f"{g:.3f}" for g in gamma_values],
                       annot=True, fmt='.1f', cmap='viridis',
                       cbar_kws={'label': 'Mean Return'})
            plt.xlabel('Epsilon')
            plt.ylabel('Gamma')
            plt.title(f'Grid Search Heatmap - {layout} (α={alpha_values[0]:.3f})')
        else:
            # Fallback to sorted bar chart
            subset_sorted = sorted(subset, key=lambda x: x["mean_return"], reverse=True)
            top_n = min(30, len(subset_sorted))  # Show only top 30
            
            plt.figure(figsize=(15, 8))
            x_pos = np.arange(top_n)
            scores = [r["mean_return"] for r in subset_sorted[:top_n]]
            
            plt.bar(x_pos, scores)
            plt.xlabel('Parameter Combination (sorted by performance)')
            plt.ylabel('Mean Return')
            plt.title(f'Top {top_n} Grid Search Results - {layout}')
            
            # Add text annotations for top performers
            for i in range(min(5, top_n)):
                r = subset_sorted[i]
                plt.text(i, scores[i] + max(scores) * 0.01,
                        f"ε={r['eps']:.3f}\nγ={r['gamma']:.3f}\nα={r['alpha']:.3f}",
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"grid_{layout}_{brick_rows}_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
def plot_metrics(data: list, metric: str = "runtime"):
    out_dir = Path("plots_f")
    out_dir.mkdir(exist_ok=True)
    
    # Filter out entries that don't have the required field
    if metric == "runtime":
        data = [d for d in data if "seconds" in d]
        values = [d["seconds"] for d in data]
        ylabel = "seconds"
        title = "Training time per layout"
        filename = "runtime.png"
    elif metric == "steps":
        data = [d for d in data if "steps" in d]
        values = [np.mean(d["steps"]) for d in data]
        ylabel = "steps"
        title = "Average training steps per layout"
        filename = "steps.png"
    elif metric == "scores":
        data = [d for d in data if "scores" in d]
        values = [np.mean(d["scores"]) for d in data]
        ylabel = "scores"
        title = "Average training scores per layout"
        filename = "scores.png"
    elif metric == "rewards":
        data = [d for d in data if "rewards" in d]
        values = [np.mean(d["rewards"]) for d in data]
        ylabel = "rewards"
        title = "Average training rewards per layout"
        filename = "rewards.png"
    
    # Extract layouts and bricks after filtering
    layouts = [d["layout"] for d in data]
    bricks = [d["bricks"] for d in data]
    
    # Create labels
    labels = [f"{l}\n({b} bricks)" for l, b in zip(layouts, bricks)]
    
    # Debug print to check data
    print(f"Number of data points: {len(data)}")
    print(f"Number of values: {len(values)}")
    print(f"Values: {values}")
    
    plt.figure(figsize=(12, 6))
    
    # Use numeric x positions
    x_pos = np.arange(len(values))
    plt.bar(x_pos, values)
    
    # Set x-axis labels only once
    plt.xticks(x_pos, labels, rotation=45 if len(labels) > 6 else 0)
    
    plt.title(title)
    plt.xlabel("Brick layout (brick count in brackets)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()
    
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np

def plot_single_metric(values: list, metric_name: str, layout: str, bricks: int, 
                      xlabel: str = "Iteration"):
    out_dir = Path("plots_f")
    out_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(20, 8))
    
    iterations = np.arange(len(values))
    values = np.array(values)
    
    # Plot original data with low opacity
    plt.plot(iterations, values, color='lightblue', linewidth=0.5, alpha=0.3, label='Raw data')
    
    # Apply smoothing based on data size
    if len(values) > 10000:
        sigma = len(values) // 10000  
        smoothed = gaussian_filter1d(values, sigma=sigma)

        
        plt.plot(iterations, smoothed, color='darkblue', linewidth=2, label='Smoothed')
    else:
        plt.plot(iterations, values, color='darkblue', linewidth=0.5)
    
    plt.title(f"{metric_name.capitalize()} over {xlabel} - {layout} ({bricks} bricks)", fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(metric_name.capitalize(), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Better axis limits
    if len(values) > 0:
        y_margin = (np.max(values) - np.min(values)) * 0.05
        plt.ylim(np.min(values) - y_margin, np.max(values) + y_margin)
    
    plt.tight_layout()
    
    filename = f"{metric_name}_{layout}_{bricks}bricks_non_smoothed.png"
    plt.savefig(out_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
   
    # Game Variables
    actions = [-1, 0, 1]  # paddle left, stay, right
    starting_vxs = [-2, -1, 0, 1, 2]
    layouts = ["pyramid", "wall", "checker"]
    n_episodes    = 10000 
    
    # Parameter Grid
    grid = {
    'eps'  : [1.0, 0.5, 0.1],
    'gamma': [0.90, 0.95, 0.99],
    'alpha': [0.05, 0.10, 0.20],
    }

    # Results
    trajectories = {layout: {} for layout in layouts}
    runtimes = []
    grid_results = []
    steps = []
    scores = []
    rewards = []
    
    
    

    
    
    # Loop to set the experiment
    for brick_rows in [2, 4]:  # brick rows to test
        for layout in layouts:
            env = make_env(layout, brick_rows=brick_rows)  # create the env with the layout and brick rows
            bricks= conf(layout,brick_rows)["num_bricks"]
            print(layout)

            grid_records = []          # collect results for this layout
            
            for eps, gamma, alpha in product(grid['eps'], grid['gamma'], grid['alpha']): # Gives cartesian product of all cobos of grid
                agent = MC_agent(actions, state_encoder, eps=eps,
                                gamma=gamma, alpha=alpha)
            
                start_t = time.perf_counter()  # Start clock for runtime
                
                infos, returns = run_training(env, agent, n_episodes)
                
                # Capture runtime
                elapsed_t = time.perf_counter() - start_t
                
                
                # Calculate mean of last 100 episodes: the "score" of this hyperparameter and then appends to grid dict
                grid_records.append({
                "eps": eps, "gamma": gamma, "alpha": alpha,
                "mean_return": np.mean(returns[-100:]),
                "agent": agent,
                "seconds": elapsed_t,
                })
            
            # Compute best Trajectories for the layout
            best = max(grid_records, key=lambda d: d['mean_return']) # We pick the agent with the best mean return
            best_agent = best['agent']
            # We freeze the exploration
            best_agent.eps = 0.0
            
            for vx0 in starting_vxs:
                trajectories[layout][vx0] = rollout_greedy(make_env(layout, brick_rows=brick_rows),
                                                        best_agent,
                                                        vx0)       
            
            # runtimes aggregated for the bar chart
            total_secs = sum(r["seconds"] for r in grid_records)
            bricks     = conf(layout,brick_rows)["num_bricks"]
            runtimes.append({"layout": layout, "bricks": bricks, "seconds": total_secs})
            steps.append({"layout": layout, "bricks": bricks, "steps": [info["steps"] for info in infos]})
            scores.append({"layout": layout, "bricks": bricks, "scores": [info["score"] for info in infos]})
            rewards.append({"layout": layout, "bricks": bricks, "rewards": returns})

            # grid scores for the performance plot
            grid_results.extend([dict(r, layout=layout) for r in grid_records])
    
        plot_trajectories(trajectories, brick_rows=brick_rows)
        plot_grid_performance(grid_results, brick_rows=brick_rows)
        plot_grid_performance_heatmap(grid_results, brick_rows=brick_rows)

    plot_metrics(runtimes)
    plot_metrics(rewards, metric="rewards")
    plot_metrics(steps, metric="steps")
    plot_metrics(scores, metric="scores")
    
    entry = next((entry for entry in rewards if entry["layout"] == "wall" and entry["bricks"] == 16), None)
    plot_single_metric(entry["rewards"], metric_name="rewards",
                       layout="wall", bricks=16, xlabel="Episode")
    
    entry = next((entry for entry in scores if entry["layout"] == "wall" and entry["bricks"] == 16), None)
    plot_single_metric(entry["scores"], metric_name="scores",
                       layout="wall", bricks=16, xlabel="Episode")
    
    
        
        

if __name__ == "__main__":
    main()        



