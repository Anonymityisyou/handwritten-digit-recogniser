import pygame,sys
import numpy as np


pygame.init()

blue = (40,65,95)
black = (0,0,0)
gray = (240,240,240)



display = pygame.display.set_mode((1400,700))
pygame.display.set_caption('draw')

class Network:
    def __init__(self):
        self.input_size = 784
        self.hidden_layer_size = 128
        self.output_size = 10
        
        self.lr = 0.0075


        self.W1 = np.load(r"C:\Users\mathi\OneDrive\Bureau\neural_network\weight_bias\W1.npy")
        self.W2 = np.load(r"C:\Users\mathi\OneDrive\Bureau\neural_network\weight_bias\W2.npy")
        self.W3 = np.load(r"C:\Users\mathi\OneDrive\Bureau\neural_network\weight_bias\W3.npy")

    def sigmoid(self,n):
        return 1 / (1 + np.exp(-n))

    def sigmoid_derivative(self,n):
        return n*(1-n)

    def forward(self,X):
        self.layer1 = self.sigmoid(np.dot(X,self.W1))
        self.layer2 = self.sigmoid(np.dot(self.layer1,self.W2))
        self.output = self.sigmoid(np.dot(self.layer2,self.W3))
        return self.output

    def backprop(self,X,y):
        self.forward(X)
        error3 = y - self.output
        delta3 = error3*self.sigmoid_derivative(self.output)

        error2 = delta3.dot(self.W3.T)
        delta2 = error2 * self.sigmoid_derivative(self.layer2)

        error1 = delta2.dot(self.W2.T)
        delta1 = error1 * self.sigmoid_derivative(self.layer1)

        self.W3 += self.lr*self.layer2.T.dot(delta3)
        self.W2 += self.lr*self.layer1.T.dot(delta2)
        self.W1 += self.lr*X.T.dot(delta1)

class gui:
    def __init__(self):
        self.grid = np.zeros((28,28))
        self.font = pygame.font.SysFont('dejavuserif', 200)

    def background(self):
        pygame.draw.rect(display,gray,[50,100,650,500])
        pygame.draw.rect(display,gray,[100,50,550,600])
        for i in range(2):
            for j in range(2):
                pygame.draw.circle(display,gray,[100+j*550,100+i*500],50)

        pygame.draw.rect(display,gray,[750,100,600,150])
        pygame.draw.rect(display,gray,[800,50,500,250])
        for i in range(2):
            for j in range(2):
                pygame.draw.circle(display,gray,[800+j*500,100+i*150],50)

        pygame.draw.rect(display,gray,[750,450,600,150])
        pygame.draw.rect(display,gray,[800,400,500,250])
        for i in range(2):
            for j in range(2):
                pygame.draw.circle(display,gray,[800+j*500,450+i*150],50)
        
    def drawing(self):
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()

            x = ((pos[0]-50)//21)
            y = ((pos[1]-50)//21)
            pygame.draw.rect(display,black,[x*21+50,y*21+50,21,21])
            if x<28 and y<28 and x>0 and y>0:
                self.grid[y][x] = 255
            for i in range(2,-3,-1):
                if i==0:
                    continue
                n = i

                if i<0:
                    n = i*-1
                color = (255-n*50,255-n*50,255-n*50)

                if x+i<28 and x+i>0:
                    if  color[0] > self.grid[y][x+i]:
                        self.grid[y][x+i] = color[0]
                        pygame.draw.rect(display,(255-color[0],255-color[0],255-color[0]),[(x+i)*21+50,y*21+50,21,21])
                if y+i<28 and y+i>0:
                    if  color[0] > self.grid[y+i][x]:
                        self.grid[y+i][x] = color[0]
                        pygame.draw.rect(display,(255-color[0],255-color[0],255-color[0]),[x*21+50,(y+i)*21+50,21,21])
            pygame.display.flip()
    
    def print(self,value,where):
        label = self.font.render(str(value),1,black)
        if where == 'up':
            pygame.draw.rect(display,gray,[800,100,500,150])
            display.blit(label,(1000,100))
        if where == 'down':
            pygame.draw.rect(display,gray,[800,450,500,150])
            display.blit(label,(850,450))


                
    def update(self):
        
        for event in pygame.event.get():
            if event == pygame.QUIT:
                game_exit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    pygame.display.flip()
                if event.key == pygame.K_BACKSPACE:
                    self.grid = np.zeros((28,28))
                    display.fill(blue)
                    self.background()
                
               
                        
        self.drawing()


gui = gui()
net = Network()

while True:
    gui.update()

    if gui.grid.sum() > 0:
        n =net.forward(gui.grid.flatten()/255)
        gui.print(np.argmax(n),'up')
        gui.print(round(n[np.argmax(n)],3), 'down')
