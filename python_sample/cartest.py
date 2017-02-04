def create_car(self, x, y, r):
    inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
    self.car_body = pymunk.Body(1, inertia)
    self.car_body.position = x, y
    self.car_shape = pymunk.Circle(self.car_body, 25)
    self.car_shape.color = THECOLORS["green"]
    self.car_shape.elasticity = 1.0
    self.car_body.angle = r
    driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
    self.car_body.apply_impulse(driving_direction)
    self.space.add(self.car_body, self.car_shape)
