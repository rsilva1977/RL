import numpy as np
import pygame
import random
import matplotlib.pyplot as plt

# ========== CONFIGURAÇÕES ==========
GRID_SIZE = 8          # Tamanho do grid (N x N)
CELL_SIZE = 60        # Tamanho de cada célula em pixels
TRAIN_EPISODES = 10000 # Episódios de treinamento
MAX_STEPS = 200        # Máximo de passos por episódio

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
GRAY = (150, 150, 150)
DARK_BLUE = (0, 0, 150)

class GridWorld:
    """Ambiente do Grid World com obstáculos"""
    
    def __init__(self, size=GRID_SIZE):
        self.size = size
        # Obstáculos no ambiente
        self.obstacles = [(2, 2), (2, 3), (3, 2), (4, 5), (5, 4), (6, 1)]
        self.reset()

    def reset(self):
        """Reinicia o ambiente"""
        self.agent_pos = (0, 0)  # Posição inicial do agente
        self.goal_pos = (self.size-1, self.size-1)  # Objetivo
        self.steps = 0
        return self.agent_pos

    def step(self, action):
        """
        Executa uma ação:
        0 = Cima, 1 = Direita, 2 = Baixo, 3 = Esquerda
        """
        x, y = self.agent_pos
        new_x, new_y = x, y

        # Movimentação
        if action == 0: new_x = max(0, x-1)
        elif action == 1: new_y = min(self.size-1, y+1)
        elif action == 2: new_x = min(self.size-1, x+1)
        elif action == 3: new_y = max(0, y-1)

        # Verifica obstáculos
        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y  # Fica na mesma posição

        self.agent_pos = (new_x, new_y)
        self.steps += 1

        # Verifica se alcançou o objetivo
        done = (self.agent_pos == self.goal_pos)

        # Sistema de recompensas
        if done:
            reward = 10.0      # Grande recompensa
        elif (new_x, new_y) == (x, y):
            reward = -0.5      # Penalidade por bater em obstáculo
        else:
            reward = -0.1      # Pequena penalidade por passo

        return self.agent_pos, reward, done

class QLearningAgent:
    """Agente que aprende usando Q-Learning"""
    
    def __init__(self, env, learning_rate=0.2, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.9995, min_exploration=0.01):
        self.env = env
        self.learning_rate = learning_rate      # Taxa de aprendizagem (α)
        self.discount_factor = discount_factor  # Fator de desconto (γ)
        self.exploration_rate = exploration_rate  # Taxa de exploração (ε)
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Tabela Q: [x][y][ação] → valor
        self.q_table = np.zeros((env.size, env.size, 4))
        self.rewards_history = []

    def choose_action(self, state, training=True):
        """Escolhe ação usando política ε-greedy"""
        if training and random.random() < self.exploration_rate:
            return random.randint(0, 3)  # Exploração: ação aleatória
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Explotação: melhor ação

    def learn(self, state, action, reward, new_state, done):
        """Atualiza a tabela Q usando a equação de Bellman"""
        x, y = state
        new_x, new_y = new_state
        current_q = self.q_table[x, y, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[new_x, new_y])

        # Atualização Q
        self.q_table[x, y, action] += self.learning_rate * (target_q - current_q)

        # Decaimento da exploração
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * self.exploration_decay)

def train_agent(env, agent, episodes=TRAIN_EPISODES, max_steps=MAX_STEPS):
    """Treina o agente"""
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            new_state, reward, done = env.step(action)
            agent.learn(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            steps += 1

        agent.rewards_history.append(total_reward)

        if episode % 1000 == 0:
            print(f"Episódio {episode:5d} | Recompensa: {total_reward:7.2f} | "
                  f"Exploração: {agent.exploration_rate:.4f}")

def draw_arrow(screen, cell_pos, direction, size=20, color=BLUE, width=2):
    """Desenha uma seta na célula"""
    x, y = cell_pos
    center_x = x * CELL_SIZE + CELL_SIZE // 2
    center_y = y * CELL_SIZE + CELL_SIZE // 2

    if direction == 0:  # Cima
        end = (center_x, center_y - size)
        points = [(center_x, center_y - size),
                 (center_x - 7, center_y - size + 10),
                 (center_x + 7, center_y - size + 10)]
    elif direction == 1:  # Direita
        end = (center_x + size, center_y)
        points = [(center_x + size, center_y),
                 (center_x + size - 10, center_y - 7),
                 (center_x + size - 10, center_y + 7)]
    elif direction == 2:  # Baixo
        end = (center_x, center_y + size)
        points = [(center_x, center_y + size),
                 (center_x - 7, center_y + size - 10),
                 (center_x + 7, center_y + size - 10)]
    elif direction == 3:  # Esquerda
        end = (center_x - size, center_y)
        points = [(center_x - size, center_y),
                 (center_x - size + 10, center_y - 7),
                 (center_x - size + 10, center_y + 7)]

    pygame.draw.line(screen, color, (center_x, center_y), end, width)
    pygame.draw.polygon(screen, color, points)

def draw_grid(screen, env, agent, show_policy=True):
    """Desenha o grid world"""
    screen.fill(WHITE)

    # Linhas do grid
    for i in range(env.size + 1):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0),
                        (i * CELL_SIZE, env.size * CELL_SIZE), 1)
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE),
                        (env.size * CELL_SIZE, i * CELL_SIZE), 1)

    # Obstáculos
    for ox, oy in env.obstacles:
        pygame.draw.rect(screen, GRAY, (ox * CELL_SIZE, oy * CELL_SIZE,
                                        CELL_SIZE, CELL_SIZE))

    # Objetivo
    gx, gy = env.goal_pos
    pygame.draw.circle(screen, GREEN, (gx * CELL_SIZE + CELL_SIZE//2,
                                       gy * CELL_SIZE + CELL_SIZE//2),
                       CELL_SIZE//3)

    # Agente
    ax, ay = env.agent_pos
    pygame.draw.rect(screen, RED, (ax * CELL_SIZE + CELL_SIZE//4,
                                   ay * CELL_SIZE + CELL_SIZE//4,
                                   CELL_SIZE//2, CELL_SIZE//2))

    # Política (setas)
    if show_policy:
        for x in range(env.size):
            for y in range(env.size):
                if (x, y) == env.goal_pos or (x, y) in env.obstacles:
                    continue
                best_action = np.argmax(agent.q_table[x, y])
                draw_arrow(screen, (x, y), best_action, size=20, color=DARK_BLUE)

    pygame.display.flip()

def show_training_plot(agent):
    """Mostra gráfico do progresso do treinamento"""
    plt.figure(figsize=(12, 6))
    window = 100
    rewards_smooth = np.convolve(agent.rewards_history,
                                np.ones(window)/window, mode='valid')

    plt.plot(agent.rewards_history, alpha=0.3, color='blue',
             label='Recompensa por episódio')
    plt.plot(np.arange(window-1, len(agent.rewards_history)),
             rewards_smooth, color='red', linewidth=2,
             label=f'Média móvel ({window} episódios)')

    plt.title('Progresso do Treinamento - Q-Learning', fontsize=14)
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize(env, agent):
    """Visualização interativa"""
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    pygame.display.set_caption("Q-Learning Grid World")
    clock = pygame.time.Clock()
    running = True
    auto_run = False
    show_policy = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_run = not auto_run
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_p:
                    show_policy = not show_policy

        # Modo automático
        if auto_run and env.agent_pos != env.goal_pos:
            action = agent.choose_action(env.agent_pos, training=False)
            env.step(action)
            pygame.time.delay(100)

        draw_grid(screen, env, agent, show_policy)
        info = f"Q-Learning | Auto: {'ON' if auto_run else 'OFF'} | Política: {'ON' if show_policy else 'OFF'}"
        pygame.display.set_caption(info)
        clock.tick(60)

    pygame.quit()

def main():
    print("=" * 60)
    print("🎮 Q-LEARNING GRID WORLD - Demonstração de Reinforcement Learning")
    print("=" * 60)
    print(f"\n📊 Treinando para {TRAIN_EPISODES} episódios...\n")

    env = GridWorld()
    agent = QLearningAgent(env)
    train_agent(env, agent)

    print("\n✅ Treinamento concluído!")
    print(f"   Taxa final de exploração: {agent.exploration_rate:.4f}\n")

    print("📈 Mostrando gráfico de progresso...")
    #show_training_plot(agent)

    print("\n🎨 Iniciando visualização interativa...")
    print("\nControles:")
    print("  ESPAÇO  - Ativar/desativar modo automático")
    print("  R       - Reiniciar posição do agente")
    print("  P       - Alternar visualização da política")
    print("  FECHAR  - Sair do programa\n")

    visualize(env, agent)

if __name__ == "__main__":
    main()
