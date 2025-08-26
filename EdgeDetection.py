
class EdgeDetection():
    from PIL import Image, ImageColor
    
    def __init__(self):
        pass
    
    def getGreyScale(image: Image, x: int, y: int) -> int:
        import math
        px = image.load()
        pixel = list(px[x, y])
        return math.floor(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
    
    def getG(image: Image):
        import math
        gx = EdgeDetection.getGX(image)
        gy = EdgeDetection.getGY(image)
        g = []
        threshold_avg = 0
        max = 0
        for x in range(len(gx)):
            g.insert(x, [])
            for y in range(len(gx[x])):
                n = math.floor(math.sqrt(gx[x][y] * gx[x][y] + gy[x][y] * gy[x][y]))
                threshold_avg += n
                if n > max: max = n
                g[x].insert(y, n)
        threshold_avg = threshold_avg / (image.size[0] * image.size[1])
        threshold = threshold_avg
        print(f"Threshold Average: {threshold_avg}\nMax Threshold: {max}\nUsing: {threshold}")
        for x in range(len(gx)):
            for y in range(len(gx[x])):
                g[x][y] = 1 if g[x][y] >= threshold else 0
        return g
    
    def getGX(image: Image):
        g = []
        for x in range(image.size[0] - 2):
            arrX = []
            dx = x + 1
            for y in range(image.size[1] - 2):
                dy = y + 1
                arrX.insert(y, EdgeDetection.getIX(image, dx, dy))
                
            g.insert(x, arrX)
        return g
                
    def getGY(image: Image):
        g = []
        for x in range(image.size[0] - 2):
            arrX = []
            dx = x + 1
            for y in range(image.size[1] - 2):
                dy = y + 1
                arrX.insert(y, EdgeDetection.getIY(image, dx, dy))
                
            g.insert(x, arrX)
        return g       
    
    def getIX(image: Image, x: int, y: int) -> int:
        g = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        
        return EdgeDetection.getI(g, image, x, y)
    
    def getIY(image: Image, x: int, y: int) -> int:
        g = [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        
        return EdgeDetection.getI(g, image, x, y)
    
    def getI(g, image, x, y):
        lx = 0
        
        for dx in range(len(g)):
            for dy in range(len(g[dx])):
                o = g[dx][dy]
                ox = x + (dx - 1)
                oy = y + (dy - 1)
                lx += EdgeDetection.getGreyScale(image, ox, oy) * o
        
        return lx
    