import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import math
import matplotlib.pyplot as plt

# Constants
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Snake game
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, RIGHT, DOWN, LEFT])
        self.food = self.place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False

    def place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        self.steps += 1
        self.direction = action

        head = self.snake[0]
        if self.direction == UP:
            new_head = (head[0], (head[1] - 1) % GRID_HEIGHT)
        elif self.direction == RIGHT:
            new_head = ((head[0] + 1) % GRID_WIDTH, head[1])
        elif self.direction == DOWN:
            new_head = (head[0], (head[1] + 1) % GRID_HEIGHT)
        else:  # LEFT
            new_head = ((head[0] - 1) % GRID_WIDTH, head[1])

        reward = 0
        if new_head in self.snake:
            self.game_over = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self.place_food()
            else:
                self.snake.pop()

        if self.steps > 100 * len(self.snake):
            self.game_over = True

        return reward

    def get_state(self):
        head = self.snake[0]
        point_l = ((head[0] - 1) % GRID_WIDTH, head[1])
        point_r = ((head[0] + 1) % GRID_WIDTH, head[1])
        point_u = (head[0], (head[1] - 1) % GRID_HEIGHT)
        point_d = (head[0], (head[1] + 1) % GRID_HEIGHT)

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
        ]

        return np.array(state, dtype=int)

    def is_collision(self, point):
        return point in self.snake

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.clicked = False

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        slider_pos = self.rect.x + int((self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.rect(screen, WHITE, (slider_pos - 5, self.rect.y, 10, self.rect.height))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.clicked = False
        elif event.type == pygame.MOUSEMOTION:
            if self.clicked:
                mouse_x = event.pos[0]
                self.val = (mouse_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val) + self.min_val
                self.val = max(self.min_val, min(self.val, self.max_val))

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class CosineAgent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_epsilon(self):
        # Cyclical epsilon using cosine function
        cycle_length = 50  # Length of one full cycle
        min_epsilon = 0.1
        max_epsilon = 0.5
        cycle_position = self.n_games % cycle_length
        cosine_value = math.cos(2 * math.pi * cycle_position / cycle_length)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * (cosine_value + 1) / 2
        return epsilon

    def get_action(self, state):
        epsilon = self.get_epsilon()
        final_move = [0, 0, 0, 0]
        if random.random() < epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = F.mse_loss(target, pred)
        loss.backward()
        self.optimizer.step()

# Agent
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = F.mse_loss(target, pred)
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Update the train function to include visualization
def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 50))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    speed_slider = Slider(10, HEIGHT + 10, WIDTH - 20, 30, 1, 800, 150)

    #stats stuff
    all_scores = []    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # agent = CosineAgent()
    game = SnakeGame()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return all_scores
            speed_slider.handle_event(event)

        # Get old state
        state_old = game.get_state()

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward = game.step(final_move.index(1))
        state_new = game.get_state()

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game.game_over)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, game.game_over)

        # Visualize the game
        screen.fill(BLACK)

        # Draw snake
        for segment in game.snake:
            pygame.draw.rect(screen, GREEN, (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw food
        pygame.draw.rect(screen, RED, (game.food[0]*GRID_SIZE, game.food[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Display score and other information
        score_text = font.render(f'Score: {game.score}', True, WHITE)
        screen.blit(score_text, (10, 10))
        
        games_text = font.render(f'Games: {agent.n_games}', True, WHITE)
        screen.blit(games_text, (10, 50))
        
        record_text = font.render(f'Record: {record}', True, WHITE)
        screen.blit(record_text, (10, 90))

        speed_slider.draw(screen)
        speed_text = font.render(f'Speed: {int(speed_slider.val)}', True, WHITE)
        screen.blit(speed_text, (WIDTH - 150, HEIGHT + 15))

        pygame.display.flip()
        clock.tick(int(speed_slider.val))  # Adjust the speed of visualization

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if game.game_over:
            
            if game.score > record:
                record = game.score

            print('[-] Game', agent.n_games, 'Score', game.score, 'Record:', record)


            # Train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()

            plot_scores.append(game.score)
            total_score += game.score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            all_scores.append(game.score)
            if agent.n_games % 50 == 0:
                plot_analysis(all_scores)

            game.reset()

def plot_analysis(scores):
    plt.figure(figsize=(12, 6))
    
    # Create sets of 50 games
    sets = [scores[i:i+50] for i in range(0, len(scores), 50)]
    
    # Create box plot
    plt.boxplot(sets, labels=[f'{i*50+1}-{(i+1)*50}' for i in range(len(sets))])
    
    plt.title('dl q-learning performance analysis')
    plt.xlabel('Games')
    plt.ylabel('Score')
    
    # Set y-axis limits to a reasonable range for Snake game scores
    plt.ylim(0, max(max(scores) + 10, 50))  # Set upper limit to max score + 10 or at least 50
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f'performance_analysis_{len(scores)}_games.png')
    plt.close()


if __name__ == '__main__':
    train()  # This will now train and visualize simultaneously
