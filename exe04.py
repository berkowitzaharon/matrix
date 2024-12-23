from math import sqrt
from PIL import Image ,ImageDraw
import numpy as np

def convert_to_grayscale(image):
    for i, row in enumerate (image):
        for j , cell in enumerate (row):
            R,G,B = cell
            gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            image[i][j] = int(gray)
    return image

class PixelNode:
    def __init__(self, x, y, color=None):
        self.x = x
        self.y = y
        self.color = color
        self.distance = float('inf') 
        self.visited = False
        self.heap_index = None
        self.father = None

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def _distance(self,other):
        x1 , y1 = self.x , self.y 
        x2 , y2 = other.x ,other.y
        return sqrt(((x1-x2)**2)+((y1-y2)**2))  
   
class PriorityQueue:
    @staticmethod
    def left_child(i):
        return (i*2)+1

    @staticmethod
    def right_child(i):
        return (i*2)+2

    @staticmethod
    def father(i):
        return (i-1)//2    
  
    def __init__(self):
        self.queue = []

    def insert(self, node):
        self.queue.append(node)
        i_node = len(self.queue)-1
        node.heap_index = i_node
        self.heapfy(i_node)
         
    def heapfy(self, i):
        i_father = PriorityQueue.father(i)
        while i>0 and self.queue[i]<self.queue[i_father]:
            self.queue[i_father].heap_index,self.queue[i].heap_index = i,i_father
            self.queue[i_father],self.queue[i]= self.queue[i],self.queue[i_father]
            i = i_father  
            i_father = PriorityQueue.father(i)
        return i

    def extract_min(self):
        if not self.queue:
            return
        self.queue[0],self.queue[-1] = self.queue[-1],self.queue[0]
        self.queue[0].heap_index = 0
        _min = self.queue.pop()
        self.heapfy_d(0)
        return _min

    def heapfy_d (self,i):
        heap_len = len(self.queue)
        left_child = PriorityQueue.left_child(i)
        while left_child < heap_len:
            _minimun = i
            if self.queue[i] > self.queue[left_child]:
                _minimun = left_child
            right_child = PriorityQueue.right_child(i)
            if right_child < heap_len and self.queue[right_child] < self.queue[_minimun]:
                _minimun = right_child
            if _minimun == i:
                break
            self.queue[i].heap_index , self.queue[_minimun].heap_index = _minimun , i   
            self.queue[i] , self.queue[_minimun] = self.queue[_minimun] , self.queue[i]   
            i = _minimun
            left_child = PriorityQueue.left_child(i)
        return i

    def decrease_key(self, node, new_distance):
        node.distance = new_distance
        node_i = node.heap_index
        node_i = self.heapfy(node_i)
        self.heapfy_d(node_i)

def dijkstra(gray_image_matrix, start, end,BOUNDARY = 200):
    pixel_nodes_matrix = [
                        [PixelNode(x,y,cell) for x, cell in enumerate(row)]
                        for y, row in enumerate(gray_image_matrix)
                          ]
    x,y = start
    start_node = pixel_nodes_matrix [y][x]
    start_node.distance = 0
    start_node.visited = True    
    queue = PriorityQueue()
    for row in pixel_nodes_matrix:
        for cell in row:
            queue.insert(cell)
    while queue.queue:
        nimmun = queue.extract_min()
        nimmun.visited = True
        x,y = nimmun.x ,nimmun.y
        if (x,y) == end:
            break        
        nimmun_neighbors = [pixel_nodes_matrix[i][j]
                            for i in range (y-1,y+2)
                            for j in range (x-1,x+2)
                            if 0<=i<len(pixel_nodes_matrix)and 0<=j<len(pixel_nodes_matrix[i])
                            and (x,y)!= (i,j) 
                            and not pixel_nodes_matrix[i][j].visited 
                            and pixel_nodes_matrix[i][j].color>BOUNDARY]
        for neighbor in nimmun_neighbors :
            new_distance = nimmun.distance + nimmun._distance(neighbor)

            if new_distance < neighbor.distance:
                neighbor.father = nimmun
                queue.decrease_key(neighbor,new_distance)
    if not queue.queue:
        return []
    res = []
    x,y = end
    while (x,y)!=start:
        res.append((x,y))
        father = pixel_nodes_matrix[y][x].father
        x,y = father.x,father.y
    res.append(start)
    return res[::-1]
def main():
    start = (0, 0)
    path = "./matrixes/maze9.jpg"
    img = Image.open(path, 'r')
    img = img.convert('RGB')
    rgb_array = np.array(img)
    rgb_list = rgb_array.tolist()
    gray_image_matrix = convert_to_grayscale(rgb_list)
    end = len(gray_image_matrix[-1]) - 1, len(gray_image_matrix) - 1
    short_path = dijkstra(gray_image_matrix, start, end)
    solution_path = './matrixes/maze9_1.jpg'
    img_output = Image.new('RGB', img.size, "white")
    draw = ImageDraw.Draw(img_output)
    
    for y, row in enumerate(gray_image_matrix):
        for x, value in enumerate(row):
            color = (value, value, value)
            img_output.putpixel((x, y), color)
    for x, y in short_path:
        img_output.putpixel((x, y), (255, 0, 0))
    
    img_output.save(solution_path)
    print(f"Solution saved to {solution_path}")
    return 

main()
