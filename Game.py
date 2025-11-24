import pygame
from pygame.locals import *
import cv2
import numpy as np
import mediapipe as mp
import time
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.window_width = 1200
        self.window_height = 700
        self.game_width = 800
        self.game_height = 700
        self.camera_width = 400
        self.camera_height = 350
        self.surface = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Snake Game with Hand Tracking')
        self.bg_color = (15, 15, 25)
        self.grid_color = (30, 30, 40)
        self.snake_color = (0, 255, 100)
        self.food_color = (255, 50, 50)
        self.text_color = (220, 220, 220)
        self.accent_color = (80, 200, 255)
        self.block_size = 25
        self.snake_pos = [[100, 100], [75, 100], [50, 100]]
        self.food_pos = [300, 300]
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        self.running = True
        self.game_over = False
        self.fps_controller = pygame.time.Clock()
        self.fps = 10
        self.speed_increment = 0.3
        self.cap = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=5)
        self.connection_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)
        self.hand_center_previous = None
        self.current_direction = None
        self.last_direction_change = time.time()
        self.direction_cooldown = 0.2
        self.movement_threshold = 30
        self.prev_hand_y = None
        self.prev_hand_x = None
    def spawn_food(self):
        while True:
            x = pygame.math.Vector2(
                np.random.randint(0, (self.game_width//self.block_size) - 1) * self.block_size,
                np.random.randint(0, (self.game_height//self.block_size) - 1) * self.block_size
            )
            if list(x) not in self.snake_pos:
                self.food_pos = list(x)
                break
    def process_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        h, w, c = frame.shape
        cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 2)
        self.current_direction = None
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    frame, 
                    handLms, 
                    self.mpHands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec
                )
                h, w, c = frame.shape
                landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmarks.append((px, py))
                    cv2.circle(frame, (px, py), 5, (0, 0, 255), cv2.FILLED)
                hand_x, hand_y = landmarks[0]
                cv2.circle(frame, (hand_x, hand_y), 10, (0, 255, 255), cv2.FILLED)
                if self.prev_hand_y is not None and self.prev_hand_x is not None:
                    dx = hand_x - self.prev_hand_x
                    dy = hand_y - self.prev_hand_y
                    if time.time() - self.last_direction_change > self.direction_cooldown:
                        if abs(dx) > abs(dy) and abs(dx) > self.movement_threshold:
                            if dx > 0:
                                self.current_direction = 'RIGHT'
                                if self.direction != 'LEFT':
                                    self.change_to = 'RIGHT'
                                    self.last_direction_change = time.time()
                            else:
                                self.current_direction = 'LEFT'
                                if self.direction != 'RIGHT':
                                    self.change_to = 'LEFT'
                                    self.last_direction_change = time.time()
                        elif abs(dy) > abs(dx) and abs(dy) > self.movement_threshold:
                            if dy > 0:
                                self.current_direction = 'DOWN'
                                if self.direction != 'UP':
                                    self.change_to = 'DOWN'
                                    self.last_direction_change = time.time()
                            else:
                                self.current_direction = 'UP'
                                if self.direction != 'DOWN':
                                    self.change_to = 'UP'
                                    self.last_direction_change = time.time()
                self.prev_hand_x = hand_x
                self.prev_hand_y = hand_y
                if self.current_direction:
                    text = self.current_direction
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    overlay = frame.copy()
                    cv2.rectangle(overlay, 
                                 (text_x - 10, text_y - text_size[1] - 10),
                                 (text_x + text_size[0] + 10, text_y + 10),
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    cv2.putText(frame, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, "Move hand to control snake", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    def update_direction(self):
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        elif self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        elif self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
    def update_snake(self):
        if self.direction == 'RIGHT':
            new_head = [self.snake_pos[0][0] + self.block_size, self.snake_pos[0][1]]
        elif self.direction == 'LEFT':
            new_head = [self.snake_pos[0][0] - self.block_size, self.snake_pos[0][1]]
        elif self.direction == 'UP':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] - self.block_size]
        elif self.direction == 'DOWN':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] + self.block_size]
        if (new_head[0] < 0 or new_head[0] >= self.game_width or
            new_head[1] < 0 or new_head[1] >= self.game_height or
            new_head in self.snake_pos):
            self.game_over = True
            return
        self.snake_pos.insert(0, new_head)
        if new_head[0] == self.food_pos[0] and new_head[1] == self.food_pos[1]:
            self.score += 10
            self.spawn_food()
            if self.fps < 20:
                self.fps += self.speed_increment
        else:
            self.snake_pos.pop()
    def draw_grid(self):
        for x in range(0, self.game_width, self.block_size):
            pygame.draw.line(self.surface, self.grid_color, (x, 0), (x, self.game_height))
        for y in range(0, self.game_height, self.block_size):
            pygame.draw.line(self.surface, self.grid_color, (0, y), (self.game_width, y))
    def draw_panel(self):
        panel_rect = pygame.Rect(self.game_width, 0, self.window_width - self.game_width, self.window_height)
        pygame.draw.rect(self.surface, (25, 25, 35), panel_rect)
        pygame.draw.line(self.surface, self.accent_color, 
                       (self.game_width, 0), (self.game_width, self.window_height), 3)
    def draw_game(self):
        self.surface.fill(self.bg_color)
        self.draw_grid()
        self.draw_panel()
        for i, pos in enumerate(self.snake_pos):
            if i == 0:
                pygame.draw.rect(self.surface, (0, 255, 150), 
                               (pos[0], pos[1], self.block_size, self.block_size), 0, 5)
                eye_size = 4
                if self.direction == 'RIGHT':
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + self.block_size - 7, pos[1] + 7), eye_size)
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + self.block_size - 7, pos[1] + self.block_size - 7), eye_size)
                elif self.direction == 'LEFT':
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + 7, pos[1] + 7), eye_size)
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + 7, pos[1] + self.block_size - 7), eye_size)
                elif self.direction == 'UP':
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + 7, pos[1] + 7), eye_size)
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + self.block_size - 7, pos[1] + 7), eye_size)
                elif self.direction == 'DOWN':
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + 7, pos[1] + self.block_size - 7), eye_size)
                    pygame.draw.circle(self.surface, (0, 0, 0), (pos[0] + self.block_size - 7, pos[1] + self.block_size - 7), eye_size)
            else:
                pygame.draw.rect(self.surface, self.snake_color, 
                               (pos[0], pos[1], self.block_size, self.block_size), 0, 3)
            if i > 0:
                green_val = max(80, 200 - (i * 3))
                pygame.draw.rect(self.surface, (0, green_val, 80), 
                               (pos[0]+4, pos[1]+4, self.block_size-8, self.block_size-8), 0, 2)
        pulse = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 10
        food_radius = (self.block_size//2) + pulse/10
        for i in range(3):
            size = food_radius + (3-i)*2
            alpha = 100 - i*30
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.food_color[:3], alpha), 
                             (size, size), size)
            self.surface.blit(s, (self.food_pos[0] + self.block_size//2 - size, 
                                self.food_pos[1] + self.block_size//2 - size))
        pygame.draw.circle(self.surface, self.food_color, 
                          (self.food_pos[0] + self.block_size//2, self.food_pos[1] + self.block_size//2), 
                          food_radius)
        self.draw_game_info()
        if self.game_over:
            overlay = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surface.blit(overlay, (0, 0))
            game_over_font = pygame.font.SysFont('Arial', 60, bold=True)
            game_over_text = game_over_font.render('GAME OVER!', True, (255, 50, 50))
            restart_text = self.font.render('Press R to restart or ESC to quit', True, self.text_color)
            text_width, text_height = game_over_text.get_size()
            glow_surf = pygame.Surface((text_width + 20, text_height + 20), pygame.SRCALPHA)
            for i in range(10, 0, -2):
                color = (255, 50, 50, i * 10)
                pygame.draw.rect(glow_surf, color, (10-i, 10-i, text_width + i*2, text_height + i*2), 0, 10)
            self.surface.blit(glow_surf, (self.game_width//2 - text_width//2 - 10, self.game_height//2 - text_height//2 - 40))
            self.surface.blit(game_over_text, (self.game_width//2 - text_width//2, self.game_height//2 - text_height//2 - 30))
            self.surface.blit(restart_text, (self.game_width//2 - restart_text.get_width()//2, self.game_height//2 + 40))
    def draw_game_info(self):
        title_text = self.title_font.render('SNAKE GAME', True, self.accent_color)
        subtitle_text = self.font.render('with Hand Tracking', True, self.text_color)
        panel_center_x = self.game_width + (self.window_width - self.game_width) // 2
        self.surface.blit(title_text, (panel_center_x - title_text.get_width()//2, 20))
        self.surface.blit(subtitle_text, (panel_center_x - subtitle_text.get_width()//2, 60))
        score_bg = pygame.Rect(panel_center_x - 100, 100, 200, 50)
        pygame.draw.rect(self.surface, (40, 40, 60), score_bg, 0, 10)
        pygame.draw.rect(self.surface, self.accent_color, score_bg, 2, 10)
        score_text = self.font.render(f'Score: {self.score}', True, self.text_color)
        self.surface.blit(score_text, (panel_center_x - score_text.get_width()//2, 110))
        dir_bg = pygame.Rect(panel_center_x - 100, 160, 200, 50)
        pygame.draw.rect(self.surface, (40, 40, 60), dir_bg, 0, 10)
        pygame.draw.rect(self.surface, self.accent_color, dir_bg, 2, 10)
        dir_text = self.font.render(f'Direction: {self.direction}', True, self.text_color)
        self.surface.blit(dir_text, (panel_center_x - dir_text.get_width()//2, 170))
        speed_text = self.font.render(f'Speed: {self.fps:.1f}', True, self.text_color)
        self.surface.blit(speed_text, (panel_center_x - speed_text.get_width()//2, 220))
        controls_y = self.camera_height + 380
        controls_title = self.font.render('Controls:', True, self.accent_color)
        self.surface.blit(controls_title, (self.game_width + 20, controls_y))
        controls = [
            "• Move your hand LEFT/RIGHT/UP/DOWN",
            "• Press arrow keys as alternative",
            "• Press R to restart after game over",
            "• Press ESC to quit"
        ]
        for i, text in enumerate(controls):
            ctrl_text = self.font.render(text, True, self.text_color)
            self.surface.blit(ctrl_text, (self.game_width + 20, controls_y + 30 + i*25))
    def draw_camera_feed(self, frame):
        frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        camera_x = self.game_width + (self.window_width - self.game_width - self.camera_width) // 2
        camera_y = 280
        camera_rect = pygame.Rect(camera_x - 3, camera_y - 3, self.camera_width + 6, self.camera_height + 6)
        pygame.draw.rect(self.surface, self.accent_color, camera_rect, 3, 5)
        self.surface.blit(frame, (camera_x, camera_y))
        camera_title = self.font.render('Hand Tracking Camera', True, self.accent_color)
        self.surface.blit(camera_title, (camera_x + (self.camera_width - camera_title.get_width())//2, camera_y - 30))
    def process_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_RIGHT and self.direction != 'LEFT':
                    self.change_to = 'RIGHT'
                elif event.key == K_LEFT and self.direction != 'RIGHT':
                    self.change_to = 'LEFT'
                elif event.key == K_UP and self.direction != 'DOWN':
                    self.change_to = 'UP'
                elif event.key == K_DOWN and self.direction != 'UP':
                    self.change_to = 'DOWN'
                elif event.key == K_r and self.game_over:
                    self.snake_pos = [[100, 100], [75, 100], [50, 100]]
                    self.direction = 'RIGHT'
                    self.change_to = self.direction
                    self.score = 0
                    self.game_over = False
                    self.fps = 10
                    self.spawn_food()
    def run(self):
        self.spawn_food()
        while self.running:
            self.process_events()
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_hand(frame)
            if not self.game_over:
                self.update_direction()
                self.update_snake()
            self.draw_game()
            self.draw_camera_feed(processed_frame)
            pygame.display.flip()
            self.fps_controller.tick(self.fps)
        self.cap.release()
        pygame.quit()
if __name__ == '__main__':
    game = SnakeGame()
    game.run()
