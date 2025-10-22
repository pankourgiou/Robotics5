import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

# ---------------- Environment Setup ----------------
dt = 0.1
steps = 800
bounds = [0, 20, 0, 12]

n_robots = 4
n_targets = 2

# Robot and target parameters
robot_speed = 0.8
sensor_range = 5.0
comm_range = 7.0
process_noise = 0.1
obs_noise = 0.5

# Initialize robot and target states
robots = [np.array([np.random.uniform(2, 18), np.random.uniform(2, 10)]) for _ in range(n_robots)]
targets = [np.array([np.random.uniform(3, 17), np.random.uniform(3, 9)]) for _ in range(n_targets)]
target_vels = [np.random.uniform(-0.3, 0.3, size=2) for _ in range(n_targets)]

# Kalman filter states for each robot-target pair
estimates = np.zeros((n_robots, n_targets, 2))  # position estimates
covariances = np.array([[np.eye(2) for _ in range(n_targets)] for _ in range(n_robots)], dtype=float)

# ---------------- Kalman Filter Functions ----------------
def kalman_predict(mu, Sigma):
    """Prediction step."""
    F = np.eye(2)
    Q = process_noise * np.eye(2)
    mu_pred = F @ mu
    Sigma_pred = F @ Sigma + Q
    return mu_pred, Sigma_pred

def kalman_update(mu, Sigma, z):
    """Update step with observation z."""
    H = np.eye(2)
    R = (obs_noise ** 2) * np.eye(2)
    y = z - H @ mu
    S = H @ Sigma @ H.T + R
    K = Sigma @ H.T @ np.linalg.inv(S)
    mu_new = mu + K @ y
    Sigma_new = (np.eye(2) - K @ H) @ Sigma
    return mu_new, Sigma_new

# ---------------- Simulation Step Functions ----------------
def step_targets():
    """Move targets and bounce off boundaries."""
    for i in range(n_targets):
        targets[i] += target_vels[i] * dt
        for d in range(2):
            if targets[i][d] < bounds[d*2] + 1 or targets[i][d] > bounds[d*2+1] - 1:
                target_vels[i][d] *= -1

def step_robots():
    """Each robot senses, communicates, and moves."""
    global estimates, covariances

    new_estimates = np.zeros_like(estimates)
    new_covs = np.zeros_like(covariances)

    # Predict & update with observations
    for i in range(n_robots):
        for j in range(n_targets):
            mu_pred, Sigma_pred = kalman_predict(estimates[i, j], covariances[i, j])
            dist = np.linalg.norm(robots[i] - targets[j])
            if dist < sensor_range:
                z = targets[j] + np.random.normal(0, obs_noise, 2)
                mu_upd, Sigma_upd = kalman_update(mu_pred, Sigma_pred, z)
            else:
                mu_upd, Sigma_upd = mu_pred, Sigma_pred
            new_estimates[i, j] = mu_upd
            new_covs[i, j] = Sigma_upd

    # Communication: share with neighbors
    for i in range(n_robots):
        neighbors = [k for k in range(n_robots) if i != k and np.linalg.norm(robots[i] - robots[k]) < comm_range]
        for j in neighbors:
            new_estimates[i] = 0.5 * (new_estimates[i] + new_estimates[j])
            new_covs[i] = 0.5 * (new_covs[i] + new_covs[j])

    estimates = new_estimates
    covariances = new_covs

    # Move toward mean of target estimates
    for i in range(n_robots):
        mean_est = np.mean(estimates[i], axis=0)
        dir_vec = mean_est - robots[i]
        dist = np.linalg.norm(dir_vec)
        if dist > 1e-3:
            robots[i] += (robot_speed * dt) * dir_vec / dist

        # Keep inside world bounds
        robots[i][0] = np.clip(robots[i][0], bounds[0]+0.5, bounds[1]-0.5)
        robots[i][1] = np.clip(robots[i][1], bounds[2]+0.5, bounds[3]-0.5)

# ---------------- Visualization ----------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[2], bounds[3])
ax.set_aspect('equal')
ax.set_title("Cooperative Multi-Robot Target Tracking (Kalman + Communication)")

robot_scat = ax.scatter([], [], s=100, facecolors='none', edgecolors='black', label="Robots")
target_scat = ax.scatter([], [], s=80, label="Targets")
ellipses = [Ellipse((0, 0), 0.1, 0.1, fill=False) for _ in range(n_targets)]
for e in ellipses:
    ax.add_patch(e)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

def init():
    robot_scat.set_offsets(np.empty((0, 2)))
    target_scat.set_offsets(np.empty((0, 2)))
    for e in ellipses:
        e.set_visible(False)
    time_text.set_text('')
    return [robot_scat, target_scat, *ellipses, time_text]

def update(frame):
    step_targets()
    step_robots()

    robot_scat.set_offsets(np.array(robots))
    target_scat.set_offsets(np.array(targets))

    # Draw uncertainty ellipses
    for j in range(n_targets):
        mu_avg = np.mean(estimates[:, j, :], axis=0)
        cov_avg = np.mean(covariances[:, j, :, :], axis=0)
        vals, vecs = np.linalg.eigh(cov_avg)
        vals = np.maximum(vals, 1e-6)
        width, height = 2.0 * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ellipses[j].set_center(mu_avg)
        ellipses[j].width = width
        ellipses[j].height = height
        ellipses[j].angle = angle
        ellipses[j].set_visible(True)

    time_text.set_text(f"t = {frame * dt:.1f} s")
    return [robot_scat, target_scat, *ellipses, time_text]

anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=40)
plt.show()
