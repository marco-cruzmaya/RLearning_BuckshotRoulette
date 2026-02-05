class Item:
    def __init__(self,type=0,pos=-1):
          self.type = type
          self.pos = pos
    
    def get_type(self):
        return self.type
    
    def get_pos(self):
        return self.pos

    def set_type(self,type):
        self.type = type
    
    def set_pos(self,pos):
        self.pos = pos