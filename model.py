import torch
from torch.distributions import Categorical
from torch.nn import Module, Linear, LeakyReLU
import numpy as np
from tqdm import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNetwork(Module):
    def __init__(self, input_size, output_size, optimizer, lr):
        super(ActorNetwork, self).__init__()
        self.output_size = output_size
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, output_size)

        self.l_relu = LeakyReLU(0.1)
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = torch.softmax(self.fc4(x), dim=-1)

        return y
    
    def select_action(self, state):
        y = self(state)
        prediction = Categorical(y)
        action = prediction.sample()

        return action.item(), prediction.probs.tolist()

    def actor_loss(self, y_true, y_pred):
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        advantages, prediction_picks, actions = y_true[0], y_true[1], y_true[2]
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = torch.clamp(prob, min=1e-10, max=1.0)
        old_prob = torch.clamp(old_prob, min=1e-10, max=1.0)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))
        p1 = ratio * advantages
        p2 = torch.clamp(ratio, min=1-LOSS_CLIPPING, max=1+LOSS_CLIPPING) * advantages

        actor_loss = -torch.mean(torch.min(p1, p2))

        entropy = -(y_pred * torch.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * torch.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def fit(self, epoch):
        self.optimizer.zero_grad()
        y_pred = self(epoch.states)
        y_true = [self.advantages, self.predictions, self.actions]
        loss = self.actor_loss(y_true, y_pred)
        loss.backward()
        self.optimizer.step()
        return loss
    
    

class CriticNetwork(Module):
    def __init__(self, input_size, optimizer, lr):
        super(CriticNetwork, self).__init__()
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, 1)

        self.l_relu = LeakyReLU(0.1)
        self.optimizer = optimizer(self.parameters(), lr=lr)
        
    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        return y.squeeze(1)
    
    def predict(self, state):            
        y = self(state)
        y = torch.softmax(y, dim=-1)
        prediction = Categorical(y).probs.tolist()

        return prediction
    
    def critic_loss(self, y_pred, values, targets):
        LOSS_CLIPPING = 0.2
        values = torch.from_numpy(np.array(values)).float().to(device)
        clipped_value_loss = values + torch.clamp(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
        v_loss1 = (targets - clipped_value_loss) ** 2
        v_loss2 = (targets - y_pred) ** 2
        value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
        return value_loss

    def fit(self, epoch, values):
        self.optimizer.zero_grad()
        y_pred = self(epoch.states)
        loss = self.critic_loss(y_pred, values, epoch.targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
class PPOAgent:
    def __init__(self, env):
        self.log_dir = "logs"
        self.env_name = "LunarLander-v2"
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]    #8
        self.action_dim = self.env.action_space.n   #4
        self.lr = 1e-3
        self.episode_ite = 0

        self.n_epochs = 100
        self.n_episodes = 10
        self.n_steps = 500
        
        self.writer = SummaryWriter(log_dir=self.log_dir, filename_suffix=self.env_name)    #logs
        
        self.actor_network = ActorNetwork(self.state_dim, self.action_dim, torch.optim.Adam, self.lr)
        self.critic_network = CriticNetwork(self.state_dim, torch.optim.Adam, self.lr)


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)


    def calc_loss(self, epoch):
        probs = self.actor_network.forward(epoch.states).gather(1, epoch.actions)
        old_probs = self.old_policy_net.forward(epoch.states).gather(1, epoch.actions)
        ratio = probs / old_probs

        surrogate_loss = torch.min(ratio * epoch.rewards, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * epoch.rewards)
        loss = -surrogate_loss.mean()
        return loss


    def update_networks(self, epoch):
        values = self.critic_network.predict(epoch.states)
        next_values = self.critic_network.predict(epoch.next_states)

        advantages, target = self.get_gaes(epoch.rewards, epoch.dones, values, next_values)
        epoch.update_epoch(advantages, target)

        actor_loss = self.actor_network.fit(epoch)
        critic_loss = self.critic_network.fit(epoch, values)

        self.writer.add_scalar('Actor_loss', actor_loss, self.episode_ite)
        self.writer.add_scalar('Critic_loss', critic_loss, self.episode_ite)

    
    def run(self):
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            epoch = Epoch()
            for i_episode in range(self.n_episodes):
                state, _ = self.env.reset()
                score = 0
                episode_steps = None
                for step in range(self.n_steps):
                    state = torch.from_numpy(state).float().to(device)

                    action, prediction = self.actor_network.select_action(state)
                    #value = self.critic_network.predict(state)
                    next_state, reward, done, *_ = self.env.step(action)

                    epoch.append(state, action, reward, next_state, done, prediction)
                    state = next_state
                    score += reward
                    if done: 
                        episode_steps = step
                        break
                self.episode_ite += 1
                if episode_steps == None: episode_steps = self.n_steps
                self.writer.add_scalar(
                    "Episode score",
                    score,
                    self.episode_ite,
                )
                self.writer.add_scalar(
                    "Episode lenght",
                    episode_steps,
                    self.episode_ite,
                )        

            epoch.end_epoch()
            self.update_networks(epoch)
            

class Epoch:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.predictions = []
        self.advantages = []
        self.targets = []

    def append(self, state, action, reward, next_state, done, prediction):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.predictions.append(prediction)

    def end_epoch(self):
        self.states = torch.from_numpy(np.vstack(self.states)).float().to(device)
        self.next_states = torch.from_numpy(np.vstack(self.next_states)).float().to(device)
        self.actions = torch.from_numpy(np.vstack(self.actions)).float().to(device)
        self.predictions = torch.from_numpy(np.vstack(self.predictions)).float().to(device)
    
    def update_epoch(self, advantages, targets):
        self.advantages = torch.from_numpy(advantages).float().to(device)
        self.targets = torch.from_numpy(targets).float().to(device)
    
    

