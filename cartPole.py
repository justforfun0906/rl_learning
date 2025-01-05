import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 如果是 gymnasium 則 import gymnasium as gym
# 若 gym 在 Python 3.11 可能有版本相容問題，可以用 pip install gym==0.26.2
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, action_dim=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 輸出 logits (對應 action_dim=2)
        logits = self.fc3(x)
        return logits
    
    def get_action(self, state):
        """
        給定單一狀態 (shape: (4,))，
        輸出一個 action 與 log_prob。
        """
        # 轉成 batch=1 的張量
        state = torch.FloatTensor(state).unsqueeze(0)  # shape (1, 4)
        logits = self.forward(state)  # shape (1, 2)

        # 用 Categorical 分佈做采樣
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()          # 得到一個整數 0 or 1
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
def discount_rewards(rewards, gamma=0.99):
    """
    給定一串 step 的 reward，例如 [r0, r1, r2, ...]，
    回傳每個 time step t 對應的折扣後回報 G_t。
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted
def run_episode(env, policy_net, gamma=0.99):
    """
    跑一個 episode，收集 (log_prob, reward)。
    回傳：
      - log_probs: list of log_prob (tensor)
      - rewards: list of float
      - total_reward: episode 最後得到的累積 reward (評估用)
    """
    log_probs = []
    rewards = []
    total_reward = 0
    
    state = env.reset()[0]  # 若是 gymnasium，env.reset() 回傳 (obs, info)
    done = False
    
    while not done:
        action, log_prob = policy_net.get_action(state)
        
        # 執行動作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 紀錄
        log_probs.append(log_prob)
        rewards.append(reward)
        
        total_reward += reward
        
        state = next_state
    
    # 計算折扣後回報
    discounted_r = discount_rewards(rewards, gamma)  # shape (episode_length,)
    
    return log_probs, discounted_r, total_reward
def train_cartpole(
    max_episodes=1000, 
    gamma=0.9, 
    lr=1e-3, 
    hidden_dim=128
):
    env = gym.make("CartPole-v1")
    
    # 環境狀態維度=4，動作維度=2
    policy_net = PolicyNetwork(state_dim=4, hidden_dim=hidden_dim, action_dim=2)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    for episode in range(max_episodes):
        log_probs, discounted_r, total_reward = run_episode(env, policy_net, gamma)
        
        # 準備計算 policy gradient loss
        # Σ_t [ -log_pi(a_t|s_t) * G_t ]
        # G_t 就是 discounted_r[t]
        loss = 0
        for log_prob, Gt in zip(log_probs, discounted_r):
            loss += -log_prob * Gt

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 顯示訓練進度
        print(f"Episode {episode}, Reward = {total_reward}")
        
        # 如果總分連續多次都達到滿分(500)，可以提早結束
        if total_reward >= 500: 
            print("Solved CartPole!")
            break
    
    env.close()
    return policy_net
def play_cartpole(env, policy_net, render=True):
    state = env.reset()[0]
    done = False
    total_reward = 0
    
    while not done:
        if render:
            env.render()
        
        action, _ = policy_net.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    return total_reward

if __name__ == "__main__":
    policy_net = train_cartpole(max_episodes=1000)
    
    # 測試
    env = gym.make("CartPole-v1", render_mode="human")  # gym 0.26+ 需要指定 render_mode="human"
    score = play_cartpole(env, policy_net, render=True)
    print("Test Score:", score)
    env.close()
