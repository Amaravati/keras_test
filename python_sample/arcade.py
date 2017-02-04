def animate(self, delta_time):
    start_time = timeit.default_timer()
 
    self.ticks_to_next_ball -= 1
    if self.ticks_to_next_ball <= 0:
        self.ticks_to_next_ball = 20
        mass = 0.5
        radius = 15
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(0, SCREEN_WIDTH)
        y = SCREEN_HEIGHT
        body.position = x, y
        shape = pymunk.Circle(body, radius, pymunk.Vec2d(0, 0))
        shape.friction=0.3
        self.space.add(body, shape)
 
        sprite = CircleSprite("images/coin_01.png", shape)
        self.sprite_list.append(sprite)
        self.ball_list.append(sprite)
 
    # Check for balls that fall off the screen
    for ball in self.ball_list:
        if ball.pymunk_shape.body.position.y < 0:
            # Remove balls from physics space
            self.space.remove(ball.pymunk_shape, ball.pymunk_shape.body)
            # Remove balls from physics list
            ball.kill()
 
    # Update physics
    self.space.step(1 / 80.0)
 
    # Move sprites to where physics objects are
    for ball in self.ball_list:
        ball.center_x = ball.pymunk_shape.body.position.x
        ball.center_y = ball.pymunk_shape.body.position.y
        ball.angle = math.degrees(ball.pymunk_shape.body.angle)
 
    self.time = timeit.default_timer() - start_time
