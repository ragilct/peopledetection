class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.pos=0
        self.dir=0
        self.prepos=0 
    def getId(self):
        return self.id
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getlast(self):
        return self.pos
    def getDir(self): 
        return self.dir
    def updateCoords(self, newX, newY,newl):
        self.x = newX
        self.y = newY
        self.prepos= self.pos
        self.pos = newl
    def updateDir(self,dir):
        self.dir=dir
