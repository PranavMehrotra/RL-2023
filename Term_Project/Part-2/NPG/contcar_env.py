import time
import pygame
import math
import numpy as np

REWARD_MULTIPLIER = 1000

class ContinuousCarRadarEnv:
    def __init__(self, window_size=(900, 900), radius=300, path_thickness=100, outer_ring_radius = 400, inner_ring_radius=150, car_length=30, car_width=20, car_speed=1., num_rays=5, ray_lengths=[60, 60, 60, 60, 60], ray_angles=[math.pi / 4, math.pi / 8, 0, -math.pi / 8, -math.pi / 4], reward_inside_path=1, reward_outside_path=-1):
        self.window_size = window_size
        self.num_actions = 3  # Steer left, go straight, steer right
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Continuous Car Control with Radar')

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.brown = (139, 69, 19)
        self.red = (255, 0, 0)

        # Circular path parameters
        self.center = (self.window_size[0] // 2, self.window_size[1] // 2)
        self.radius = radius
        self.outer_ring_radius = outer_ring_radius
        self.inner_ring_radius = inner_ring_radius
        self.path_thickness = path_thickness
        self.inner_radius = self.radius - self.path_thickness

        # Car parameters
        self.car_length, self.car_width, self.car_speed = car_length, car_width, car_speed
        self.car_angle = math.pi / 2  # Start facing upward
        self.car_x, self.car_y = self.center[0] + self.radius - self.path_thickness/2, self.center[1]

        # Radar parameters
        # Check if ray_lengths and ray_angles are of the same length and equal to num_rays
        if len(ray_lengths) != len(ray_angles) or len(ray_lengths) != num_rays:
            raise Exception("ray_lengths and ray_angles must have length = num_rays")
        self.num_rays, self.ray_lengths, self.ray_angles = num_rays, np.array(ray_lengths), np.array(ray_angles)
        # self.ray_values = np.zeros(self.num_rays) # Initialize radar values to 0

        # Rewards and score
        self.reward_inside_path = reward_inside_path
        self.reward_outside_path = reward_outside_path
        self.score = 0
        
        # Checkpoints
        self.quarter_lap_done = False
        self.half_lap_done = False
        self.three_quarter_lap_done = False

        # Observation space
        self.observation = np.zeros(self.num_rays + 3) # 3 extra values for car_x, car_y, car_angle
        # Update observation
        self.observation[-3:] = [self.car_x, self.car_y, self.car_angle]
        # Initialize Pygame clock
        self.clock = pygame.time.Clock()

        # Initialize the environment
        self.reset()

    def set_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        # Take an action (change car angle)
        if action == 0:  # Steer left
            self.car_angle += 0.1
        elif action == 1:  # Go straight
            pass
        elif action == 2:  # Steer right
            self.car_angle -= 0.1

        # Calculate new car position
        update_x = self.car_speed * math.cos(self.car_angle)
        update_y = -(self.car_speed * math.sin(self.car_angle))

        # Add stochastic wind effect
        update_x += np.random.normal(0, 0.2)
        update_y += np.random.normal(0, 0.2)


        self.car_x += update_x
        self.car_y += update_y

        # # Wrap car position around the circular path
        # self.car_x %= self.window_size[0]
        # self.car_y %= self.window_size[1]
        # If car touches the outer ring, reflect it back
        distance_to_center = math.sqrt((self.car_x - self.center[0]) ** 2 + (self.car_y - self.center[1]) ** 2)
        if distance_to_center > self.outer_ring_radius:
            self.car_x -= update_x
            self.car_y -= update_y
            self.car_angle += math.pi
        elif distance_to_center < self.inner_ring_radius:
            self.car_x -= update_x
            self.car_y -= update_y
            self.car_angle += math.pi
        
        # Accomodate car_angle in [0, 2*pi]
        if self.car_angle > 2 * math.pi:
            self.car_angle -= 2 * math.pi
        elif self.car_angle < 0:
            self.car_angle += 2 * math.pi

        # Check if the car is inside the circular path
        # distance_to_center = math.sqrt((self.car_x - self.center[0]) ** 2 + (self.car_y - self.center[1]) ** 2)
        if self.inner_radius < distance_to_center < self.radius:
            reward = self.reward_inside_path
        else:
            reward = self.reward_outside_path

        # Cast radar rays and update ray_values
        # Inefficient way
        # for i in range(self.num_rays):
        #     ray_x = self.car_x + self.ray_lengths[i] * math.cos(self.car_angle + self.ray_angles[i])
        #     ray_y = self.car_y - self.ray_lengths[i] * math.sin(self.car_angle + self.ray_angles[i])

        #     distance_to_center = math.sqrt((ray_x - self.center[0]) ** 2 + (ray_y - self.center[1]) ** 2)
        #     if distance_to_center < self.radius:
        #         self.ray_values[i] = 1
        #     else:
        #         self.ray_values[i] = -1

        # Using numpy arrays for faster computation
        ray_x = self.car_x + self.ray_lengths * np.cos(self.car_angle + self.ray_angles)
        ray_y = self.car_y - self.ray_lengths * np.sin(self.car_angle + self.ray_angles)

        # Calculate distance_to_center for all rays at once
        distance_to_center = np.sqrt((ray_x - self.center[0]) ** 2 + (ray_y - self.center[1]) ** 2)

        # Update ray_values based on whether the rays are inside or outside the circular path of thickness path_thickness
        self.observation[:-3] = np.where(((distance_to_center < self.radius) & (distance_to_center > self.inner_radius)), 1, -1)


        '''
        # Update score based on radar ray values
        if all(value == 1 for value in self.ray_values):
            reward = self.reward_inside_path
        else:
            reward = self.reward_outside_path
        '''
        

        # Check if the car has reached the goal
        done = False
        if reward == self.reward_inside_path:
            # If car has crossed the finish line, done = True
            if (self.quarter_lap_done & self.half_lap_done & self.three_quarter_lap_done) == True and self.car_y < self.center[1] and self.car_x < self.center[0] + self.radius and self.car_x > self.center[0] +  self.inner_radius:
                done = True
                reward = REWARD_MULTIPLIER * 5
                print("Completed successfully")

            # Check if the car has crossed three quarters of the lap
            if self.quarter_lap_done & self.half_lap_done & (self.three_quarter_lap_done == False) and self.car_x > self.center[0] and self.car_y < self.center[1] + self.radius and self.car_y > self.center[1] +  self.inner_radius:
                self.three_quarter_lap_done = True
                reward = REWARD_MULTIPLIER * 3
            # Check if the car has crossed half the lap
            if self.quarter_lap_done == True and (self.half_lap_done | self.three_quarter_lap_done) == False and self.car_y > self.center[1] and self.car_x > self.center[0] - self.radius and self.car_x < self.center[0] -  self.inner_radius: 
                self.half_lap_done = True
                reward = REWARD_MULTIPLIER * 2
                print("Half lap done")
                print(f"Car position: ({self.car_x}, {self.car_y}), Car angle: {self.car_angle}")
            
            # Check if the car has crossed a quarter lap
            if self.quarter_lap_done == False and self.car_x < self.center[0] and self.car_y > self.center[1] - self.radius and self.car_y < self.center[1] -  self.inner_radius:
                self.quarter_lap_done = True
                reward = REWARD_MULTIPLIER * 1
        
        # Update observation and score
        self.score = reward
        self.observation[-3:] = [self.car_x, self.car_y, self.car_angle]
        return self.observation, reward, done

    def reset(self):
        # Reset car position and radar values
        self.car_x, self.car_y = self.center[0] + self.radius - self.path_thickness/2, self.center[1]
        self.car_angle = math.pi / 2
        # self.ray_values = [0] * self.num_rays
        self.score = 0
        self.done = False
        self.quarter_lap_done = False
        self.half_lap_done = False
        self.three_quarter_lap_done = False

        # Reset observation
        self.observation[:-3] = 0
        self.observation[-3:] = [self.car_x, self.car_y, self.car_angle]
        return self.observation

    def render(self):
        # Main rendering loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        # self.clock.tick(120)

        # Clear the screen
        self.screen.fill(self.white)

        # Draw circular path
        pygame.draw.circle(self.screen, self.brown, self.center, self.radius, self.path_thickness)

        # Draw Outer Ring
        pygame.draw.circle(self.screen, self.black, self.center, self.outer_ring_radius, 10)

        # Draw Inner Ring
        pygame.draw.circle(self.screen, self.black, self.center, self.inner_ring_radius, 10)

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
        font = pygame.font.Font(None, 22)
        text = font.render(f"State: {self.observation}  Score: {self.score}", True, (0, 0, 0))
        score_box.blit(text, (10, 10))
        self.screen.blit(score_box, (0, self.window_size[1] - 100))
        
        # Update the display
        pygame.display.flip()

    def close(self):
        pygame.quit()

# Initialize Pygame
# pygame.init()

# # Create environment instance
# env = ContinuousCarRadarEnv()

# # Example of using the environment
# state = env.reset()
# env.render()
# for i in range(1850):
#     if i%26 == 0:
#         action = 0
#     else:
#         action = 1
#     # action=1
#     next_state, reward, done, _ = env.step(action)
#     env.render()
#     if done:
#         print("Done")
#         break

# # Quit Pygame
# pygame.quit()
