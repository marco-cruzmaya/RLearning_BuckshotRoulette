from Game.Loadout import Loadout

class FakeLoadout(Loadout):

    def __init__(self,items=[]):
        super().__init__()
        if items != []:
            self.insert_items(items)
    
    def insert_item(self,item):
        if self.free_slots == []:
            return False
        pos = self.free_slots.pop()
        self.items[pos] = item
        return True
    
    def take_item(self,pos):
        if self.items[pos] != 0:
            item = self.items[pos]
            self.items[pos] = 0
            self.free_slots.append(pos)
            return item
        else:
            return 0
        
    def insert_items(self,items):
        if len(items) <= 8:
            self.items = items + [0]*(8-len(items))
            self.free_slots = self.free_slots[:-len(items)]
        else:
            self.items = items[:8]
            self.free_slots = []