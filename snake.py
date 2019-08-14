import numpy as np
import pygame as pg
from random import randint

class vec2:
	def __init__(self, x, y):
		self.x = int(x)
		self.y = int(y)

	def add(self, x, y):
		self.x += x
		self.y += y

	def __str__(self):
		return f'({self.x},{self.y})'

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y

class snake:
	def __init__(self, w, h):
		self.h = h
		self.w = min(w, h)
		self.cols = 20
		self.rows = 20
		self.top_padding = h-w
		self.gs = (h-self.top_padding)/self.cols

		self.score_font = pg.font.SysFont('Arial', int(self.top_padding * 0.7))
		self.reset(5)

	def reset(self, size):
		self.dir = vec2(0,-1)
		self.head = vec2(self.cols / 2 * self.gs, (self.rows / 2 * self.gs) + self.top_padding)
		self.body = [vec2(self.head.x, self.head.y+(self.gs*(size-1)))]
		for i in range(1,size-1):
			self.body.append(vec2(self.body[i-1].x, self.body[i-1].y-self.gs))
		self.food = self.create_food()
		self.score = 0
		self.delta_score = 0
		self.ended = False

	def is_ended(self):
		return self.ended

	def in_body(self, food, body):
		for v in body:
			if food == v:
				return True
		return False

	def create_food(self):
		result = vec2(randint(0,self.cols-1) * self.gs, (randint(0, self.rows-1) * self.gs) + self.top_padding)
		while self.in_body(result, self.body):
			result = vec2(randint(0,self.cols-1) * self.gs, (randint(0, self.rows-1) * self.gs) + self.top_padding)

		print("new food at: ", result)
		return result

	def eat(self):
		self.body.insert(0,vec2(self.body[0].x, self.body[0].y))
		self.food = self.create_food()

	def collided(self):
		for v in self.body:
			if (self.head == v):
				return True
		return False

	def move(self):
		self.body.pop(0)
		self.body.append(vec2(self.head.x, self.head.y))
		self.head.add(self.gs * self.dir.x, self.gs * self.dir.y)
		self.wrap()

	def wrap(self):
		if self.head.x < 0:
			self.head.x = (self.cols-1) * self.gs
		if self.head.x > (self.cols - 1) * self.gs:
			self.head.x = 0
		if self.head.y < self.top_padding:
			self.head.y = (self.rows-1) * self.gs + self.top_padding
		if self.head.y > (self.rows-1) * self.gs + self.top_padding:
			self.head.y = self.top_padding;

	def update(self):
		self.move()

		if (self.food == self.head):
			self.eat()
			self.score += 1
			self.delta_score += 1
		elif self.collided():
			self.ended = True
		
	def perform_action(self, action):
		if self.ended:
			return

		action = self.to_action_char(action)

		if action == 'L' and self.dir != vec2(1,0):
			self.dir = vec2(-1,0)
		elif action == 'R' and self.dir != vec2(-1,0):
			self.dir = vec2(1,0)
		elif action == 'U' and self.dir != vec2(0,1):
			self.dir = vec2(0,-1)
		elif action == 'D' and self.dir != vec2(0,-1):
			self.dir = vec2(0,1)

		self.update()

	def get_state(self, screen):
		pixels = pg.surfarray.array3d(screen)
		pixels = np.fliplr(np.flip(np.rot90(pixels)))
		reward = self.delta_score
		self.delta_score = 0

		return pixels, reward

	def draw(self, surf):
		pg.draw.rect(surf,(50,255,50),(self.food.x+1, self.food.y+1,self.gs-2,self.gs-2)) # food
		pg.draw.rect(surf,(200,20,200),(self.head.x+1,self.head.y+1,self.gs-2,self.gs-2)) # head

		for v in self.body:
			pg.draw.rect(surf,(200,0,0),(v.x+1,v.y+1,self.gs-2,self.gs-2)) # body

		#UI
		pg.draw.line(surf,(255,255,255,),(0,self.top_padding),(self.w,self.top_padding))
		text = self.score_font.render(f'Score: {self.score}', False, (255, 255, 255))
		surf.blit(text, (0, 0))

	def to_action_char(self, x):
		if x == 0:
			return 'L'
		elif x == 1:
			return 'R'
		elif x == 2:
			return 'U'
		elif x == 3:
			return 'D'