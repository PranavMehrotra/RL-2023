import pygame
import math
import numpy as np

class ContinuousCarRadarEnv:
    def __init__(self, window_size=(800, 900)):
        self.window_size = window_size

        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Continuous Car Control with Radar')

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.brown = (139, 69, 19)

        # Circular path parameters
        self.center = (self.window_size[0] // 2, self.window_size[1] // 2)
        self.radius = 200
        self.path_thickness = 50

        # Car parameters
        self.car_length = 30
        self.car_width = 20
        self.car_speed = 1.
        self.car_angle = math.pi / 2  # Start facing upward
        self.car_x, self.car_y = self.center[0] + self.radius - 25, self.center[1]

        # Radar parameters
        self.num_rays = 5
        self.ray_lengths = [50, 75, 100, 125, 150]
        self.ray_angles = [-math.pi / 4, -math.pi / 8, 0, math.pi / 8, math.pi / 4]
        self.ray_values = [0] * self.num_rays  # Store radar ray values (1, -1, or 0)

        # Rewards and score
        self.reward_inside_path = 1
        self.reward_outside_path = -1
        self.score = 0

        # Initialize Pygame clock
        self.clock = pygame.time.Clock()

        # Initialize the environment
        self.reset()

    def step(self, action):
        # Take an action (change car angle)
        if action == 0:  # Steer left
            self.car_angle += 0.1
        elif action == 1:  # Go straight
            pass
        elif action == 2:  # Steer right
            self.car_angle -= 0.1

        # Calculate new car position
        self.car_x += self.car_speed * math.cos(self.car_angle)
        self.car_y -= self.car_speed * math.sin(self.car_angle)

        # Wrap car position around the circular path
        self.car_x %= self.window_size[0]
        self.car_y %= self.window_size[1]

        # Cast radar rays and update ray_values
        for i in range(self.num_rays):
            ray_x = self.car_x + self.ray_lengths[i] * math.cos(self.car_angle + self.ray_angles[i])
            ray_y = self.car_y - self.ray_lengths[i] * math.sin(self.car_angle + self.ray_angles[i])

            distance_to_center = math.sqrt((ray_x - self.center[0]) ** 2 + (ray_y - self.center[1]) ** 2)
            if distance_to_center < self.radius:
                self.ray_values[i] = 1
            else:
                self.ray_values[i] = -1

        '''
        # Update score based on radar ray values
        if all(value == 1 for value in self.ray_values):
            reward = self.reward_inside_path
        else:
            reward = self.reward_outside_path
        '''
        
        # Check if the car is inside the circular path
        distance_to_center = math.sqrt((self.car_x - self.center[0]) ** 2 + (self.car_y - self.center[1]) ** 2)
        if self.radius - 50 < distance_to_center < self.radius:
            reward = self.reward_inside_path
        else:
            reward = self.reward_outside_path
        self.score = reward

        # Check if the car has reached the goal
        done = False
        if reward == self.reward_inside_path:
            if np.allclose(self.car_x, self.center[0]) and np.allclose(self.car_y, self.center[1]):
                done = True

        # Return observation, reward, done, info
        observation = np.array(self.ray_values + [self.car_x, self.car_y, self.car_angle])
        return observation, reward, done, {}

    def reset(self):
        # Reset car position and radar values
        self.car_x, self.car_y = self.center[0] + self.radius - 25, self.center[1]
        self.car_angle = math.pi / 2
        self.ray_values = [0] * self.num_rays
        self.score = 0

        # Return initial observation
        observation = np.array(self.ray_values + [self.car_x, self.car_y, self.car_angle])
        return observation

    def render(self):
        # Main rendering loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        self.clock.tick(60)

        # Clear the screen
        self.screen.fill(self.white)

        # Draw circular path
        pygame.draw.circle(self.screen, self.brown, self.center, self.radius, self.path_thickness)

        # Draw car as an arrow
        car_points = [
            (self.car_x + self.car_length * math.cos(self.car_angle), self.car_y - self.car_length * math.sin(self.car_angle)),
            (self.car_x + self.car_width * math.cos(self.car_angle - math.pi / 2), self.car_y - self.car_width * math.sin(self.car_angle - math.pi / 2)),
            (self.car_x - self.car_width * math.cos(self.car_angle), self.car_y + self.car_width * math.sin(self.car_angle)),
            (self.car_x + self.car_width * math.cos(self.car_angle + math.pi / 2), self.car_y - self.car_width * math.sin(self.car_angle + math.pi / 2))
        ]
        pygame.draw.polygon(self.screen, self.black, car_points)

        # Draw radar rays
        for i in range(self.num_rays):
            ray_x = self.car_x + self.ray_lengths[i] * math.cos(self.car_angle + self.ray_angles[i])
            ray_y = self.car_y - self.ray_lengths[i] * math.sin(self.car_angle + self.ray_angles[i])
            pygame.draw.line(self.screen, self.black, (self.car_x, self.car_y), (ray_x, ray_y), 1)

        # Draw score box
        score_box = pygame.Surface((self.window_size[0], 100))
        score_box.fill((200, 200, 200))
        font = pygame.font.Font(None, 24)
        text = font.render(f"State: {self.ray_values + [self.car_x, self.car_y, self.car_angle]}  Score: {self.score}", True, (0, 0, 0))
        score_box.blit(text, (10, 10))
        self.screen.blit(score_box, (0, self.window_size[1] - 100))
        
        # Update the display
        pygame.display.flip()


# Initialize Pygame
pygame.init()

# Create environment instance
env = ContinuousCarRadarEnv()

# Example of using the environment
state = env.reset()
env.render()
for i in range(1350):
    if i%17 == 0:
        action = 0
    else:
        action = 1
    next_state, reward, done, _ = env.step(action)
    env.render()

# Quit Pygame
pygame.quit()
