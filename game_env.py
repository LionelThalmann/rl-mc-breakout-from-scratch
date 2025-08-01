import numpy as np
import matplotlib.pyplot as plt


def conf(layout='checker',brick_rows=4):  
    
    # GRID
    grid_w = 14
    grid_h = 10
    
    # Bricks
    brick_rows = brick_rows
    brick_size_x = 3
    
    # PADDLE
    paddle_size_x = 5
    paddle_size_y = 1
    paddle_center_x = grid_w // 2
    paddle_start_y = grid_h-1
    paddle_vel_x = 0
    paddle_vel_y = 0
    
    # BALL
    ball_size_x = 1
    ball_size_y = 1 
    ball_start_pos_x = grid_w // 2
    ball_start_pos_y = grid_h - 2
    ball_vel_x = np.random.randint(-2,3)
    ball_vel_y = -1

    
    
    # Create empy grid cords
    grid = np.zeros((grid_h, grid_w), dtype=int) # because numpy is (row, colum)

    # Create the start x-coordinates for paddle
    left_x = paddle_center_x - paddle_size_x // 2
    right_x = left_x + paddle_size_x
    paddle_start_x = np.arange(left_x, right_x, dtype=int)
        
    # Create the brick coordinates according to layout
    anchors = set() # We use anchors to hold the leftmost cord of a brick (3*1)
    
    def add_anchor(x, y):
        ax = (x // brick_size_x) * brick_size_x       #  Get anchor x cord from brick index (where the brick is)
        if ax + brick_size_x <= grid_w:
            anchors.add((ax, y))

    if layout == 'wall':
        for x in range(grid_w):
            for y in range(brick_rows):
                add_anchor(x, y)

    elif layout == 'pyramid':
        for y in range(brick_rows):
            row_bricks = brick_rows - y  
            total_w = row_bricks * brick_size_x                      
            start_x = (grid_w - total_w) // 2
            for b in range(row_bricks):
                add_anchor(start_x + b * brick_size_x, y)

    elif layout == 'checker':
        for x in range(grid_w):
            for y in range(brick_rows):
                if (x // brick_size_x) % 2 == 0 and y % 2 == 0:
                    add_anchor(x, y)

    layout_out = list(anchors)
    num_bricks = len(layout_out)
    
    return {
        "grid_w" : grid_w,
        "grid_h" : grid_h,
        "grid" : grid,
        "layout" : layout_out,
        "paddle_size_x" : paddle_size_x,
        "paddle_size_y" : paddle_size_y,
        "num_bricks" : num_bricks,
        "brick_size_x":  brick_size_x,
        "paddle_start_x" : paddle_start_x,
        "paddle_start_y" : paddle_start_y,
        "ball_size_x" : ball_size_x,
        "ball_size_y" : ball_size_y,
        "ball_start_pos_x" : ball_start_pos_x,
        "ball_start_pos_y" : ball_start_pos_y,
        "ball_vx_start" : ball_vel_x,
        "ball_vy_start" : ball_vel_y,
        "paddle_vx_start" : paddle_vel_x,
        "paddle_vy_start" : paddle_vel_y,
    }


# Enviorment
class game_env:
    def __init__(self, brick_rows=4):
        # Read and instantiate all the attributes form conf 
        cfg = conf(brick_rows=brick_rows)
        self.__dict__.update(cfg)
        
        # Ball at start
        self.ball_x = self.ball_start_pos_x
        self.ball_y = self.ball_start_pos_y
        self.ball_vx = self.ball_vx_start
        self.ball_vy = self.ball_vy_start
        # Paddle at start
        self.paddle_x = self.paddle_start_x
        # Bricks recount
        self.num_bricks = len(self.layout)
        # Game over or game won flag - game needs to be rest when done == True
        self.done = False
        self.score = 0
        self.win = False
        self.paddle_speed = 0
    
        
    # Game Methods
    def reset(self):
        self.done = False
        self.score = 0
        
        # Rest Grid
        self.grid.fill(0)   
        
        # Spawn paddle in start pos
        self.paddle_x = self.paddle_start_x.copy()
        
        # spawn ball in start pos
        self.ball_x = self.ball_start_pos_x
        self.ball_y = self.ball_start_pos_y
       
        #start ball with random speed
        self.ball_vx = np.random.randint(-2,3)
        self.ball_vy = self.ball_vy_start
        
        # spawn bricks according to layout cords
        for x, y in self.layout:
            self.grid[y, x:x + self.brick_size_x] = 1  # since numpy orders row,colum and above we have coordinates (x,y) we nned to flip it
        self.num_bricks = len(self.layout)
            
        
        # Return observation (state) for agent
        obs = self.make_obs() 
        return obs
        
    def step(self, action):
        # game over or won
        if self.done:
            raise RuntimeError("Episode finished")
        
        # Set reward (-1 for every step)
        reward = -1.0
        self.paddle_speed += action
        if self.paddle_speed > 2:
            self.paddle_speed = 2
        elif self.paddle_speed < -2:
            self.paddle_speed = -2
        # move paddle if within the grid
        new_left = self.paddle_x[0] + self.paddle_speed
        if 0 <= new_left <= self.grid_w - self.paddle_size_x:
            self.paddle_x += self.paddle_speed

        hit_x = self.ball_x + self.ball_vx
        hit_y = self.ball_y + self.ball_vy 
        
        # Ball hits PADDLE =>  adjust speed depanding on where it hits
        if hit_y == self.grid_h -1 and hit_x >= self.paddle_x[0] and hit_x <= self.paddle_x[self.paddle_size_x -1]:
            if hit_x == self.paddle_x[0]:
                self.ball_vx = -2 
            if hit_x == self.paddle_x[1]:
                self.ball_vx = -1
            if hit_x == self.paddle_x[2]:
                self.ball_vx = 0
            if hit_x == self.paddle_x[3]:
                self.ball_vx = 1
            if hit_x == self.paddle_x[4]:
                self.ball_vx = 2
            self.ball_vy = -self.ball_vy # Flip direction
            
       
        # Ball hits BOTTOM
        if hit_y >= self.grid_h:
            reward += -10
            self.done = True
            return self.make_obs(), reward, self.done, {"score": self.score, "win": False}

            
        # ball hits SIDEWALL => reflect
        if hit_x >= self.grid_w -1:
            self.ball_vx = -self.ball_vx
            hit_x = self.grid_w - 1 
        elif hit_x <= 0:
            self.ball_vx = -self.ball_vx
            hit_x = 0
        
        # Ball hits ROOF => reflect
        if hit_y <= 0:
            self.ball_vy = -self.ball_vy
            hit_y = 0
        
        # Ball hits BRICK
        if 0 <= hit_x < self.grid_w and 0 <= hit_y < self.grid_h and self.grid[hit_y, hit_x]:
            anchor_x = (hit_x // self.brick_size_x) * self.brick_size_x
            self.grid[hit_y, anchor_x:anchor_x + self.brick_size_x] = 0 
            self.num_bricks -= 1
            if self.num_bricks <= 0:
                self.done = True
                self.win = True
                reward += 10
                self.score+=1
            else:
                self.score+=1
                reward+=1
            
            # Ball hits from above or below - vy reverses
            if (hit_y - self.ball_y) != 0:
                self.ball_vy = -self.ball_vy
            else:
                # Ball hits from side - vx reverses
                self.ball_vx = -self.ball_vx
          
            
        # Apply step => Move the Ball
        self.ball_x = hit_x
        self.ball_y = hit_y  
        
        obs = self.make_obs() 
        return obs, reward, self.done, {"score": self.score, "win": self.win}

    def make_obs(self):
        return {
         "ball_x": self.ball_x,
         "ball_y": self.ball_y,
         "ball_vx": self.ball_vx,
         "ball_vy": self.ball_vy,
         "paddle_x": self.paddle_x,
         "grid": self.grid,
         "num_bricks": self.num_bricks, # Is this necessary?
        }

