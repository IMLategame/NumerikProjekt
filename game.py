# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:39:40 2020

@author: Eva
"""
import pygame, sys
#from pygame.locals import *
 
# Initialize program
pygame.init()
 
# Assign FPS a value
FPS = 30
FramePerSec = pygame.time.Clock()

#Farben
schwarz = (0,0,0)   
rot = (255, 0 ,0)   
blau = (0, 0, 255) 
weiss = (255, 255, 255)   

#Brettgrößen
brett_breite = 600
brett_hohe = 600

#Brett
dispbrett = pygame.display.set_mode((brett_breite, brett_hohe))
dispbrett.fill(weiss)

#Brettlinien
pygame.draw.rect(dispbrett, schwarz, (100, 100, 400, 400), 2)
pygame.draw.rect(dispbrett, schwarz, (150, 150, 300, 300), 2)
pygame.draw.rect(dispbrett, schwarz, (200, 200, 200, 200), 2)
pygame.draw.line(dispbrett, schwarz, (300,100), (300, 200))
pygame.draw.line(dispbrett, schwarz, (100,300), (200, 300))
pygame.draw.line(dispbrett, schwarz, (300,400), (300, 500))
pygame.draw.line(dispbrett, schwarz, (400,300), (500, 300))

#Ecken
pygame.draw.circle(dispbrett, schwarz, (100, 100), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 100), 5)
pygame.draw.circle(dispbrett, schwarz, (500, 100), 5)
pygame.draw.circle(dispbrett, schwarz, (150, 150), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 150), 5)
pygame.draw.circle(dispbrett, schwarz, (450, 150), 5)
pygame.draw.circle(dispbrett, schwarz, (200, 200), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 200), 5)
pygame.draw.circle(dispbrett, schwarz, (400, 200), 5)
pygame.draw.circle(dispbrett, schwarz, (100, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (150, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (200, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (400, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (450, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (500, 300), 5)
pygame.draw.circle(dispbrett, schwarz, (200, 400), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 400), 5)
pygame.draw.circle(dispbrett, schwarz, (400, 400), 5)
pygame.draw.circle(dispbrett, schwarz, (150, 450), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 450), 5)
pygame.draw.circle(dispbrett, schwarz, (450, 450), 5)
pygame.draw.circle(dispbrett, schwarz, (100, 500), 5)
pygame.draw.circle(dispbrett, schwarz, (300, 500), 5)
pygame.draw.circle(dispbrett, schwarz, (500, 500), 5)

#Pseudocode wie es weiter gehen würde
#
#class Player(pygame.sprite.Sprite):
#    def __init__(self):
#        super().__init__() 
#       ... verknupfen mit class PlayerI?
#
#    def move(self):
#
#       raise raise NotImplementedError()  
#       ... verknupfen mit getMove?      
        
# Beginning Game Loop
while True:
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
   
    FramePerSec.tick(FPS)
