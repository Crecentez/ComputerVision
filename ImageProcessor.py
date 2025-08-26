import numpy as np
from numpy import ndarray
from PIL import Image
import math

class ImageProcessor:
    
    
    def __init__(self):
        pass
    
    def GetArr(image: Image) -> ndarray:
        return np.array(image)
        # return [[image.getpixel((x, y)) for x in range(image.size[0])] for y in range(image.size[1])]
    
    def SaveImageArr(imageArr: ndarray, path):
        # h = len(imageArr)
        # w = len(imageArr[0])
        # img = Image.new("L", (w, h))
        # for x in range(w):
        #     for y in range(h):
        #         img.putpixel((x, y), imageArr[y][x])
        img = Image.fromarray(imageArr)
        img.save(path)
        img.show()
        
    def ApplyGaussianFilter(image: ndarray, sigma=1, radius=1) -> tuple[ndarray, ndarray]:
        kernalSize = radius * 2 + 1
        
        kernal = np.zeros((kernalSize, kernalSize))
        k_sum = 0
        for kx in range(kernalSize):
            for ky in range(kernalSize):
                n = ImageProcessor.GetGuassian(kx - radius, ky - radius, sigma)
                k_sum += n
                kernal[kx, ky] = n
        for kx in range(kernalSize):
            for ky in range(kernalSize):
                kernal[kx, ky] = kernal[kx, ky] / k_sum
                
        filtered = np.zeros_like(image, dtype=np.uint8)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                val = 0
                sum = 0
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        xi = x + dx
                        yi = y + dy
                        if 0 <= xi < image.shape[0] and 0 <= yi < image.shape[1]:
                            w = kernal[dx + radius, dy + radius]
                            val += image[xi, yi] * w
                            sum += w
                filtered[x, y] = round(val / sum)
        return filtered
    
    def GetO(image: ndarray):
        sgx = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        sgy = [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        amap = [
            {
                "Min": 0,
                "Max": 22.5,
                "Map": [[0,   0,   0],
                        [255, 255, 255],
                        [0,   0,   0]]
            },
            {
                "Min": 22.5,
                "Max": 67.5,
                "Map": [[0,   0,   255],
                        [0,   255, 0],
                        [255, 0,   0]]
            },
            {
                "Min": 67.5,
                "Max": 112.5,
                "Map": [[0,   255, 0],
                        [0,   255, 0],
                        [0,   255, 0]]
            },
            {
                "Min": 112.5,
                "Max": 157.5,
                "Map": [[255, 0,   0],
                        [0,   255, 0],
                        [0,   0,   255]]
            },
            {
                "Min": 157.5,
                "Max": 180,
                "Map": [[0,   0,   0],
                        [255, 255, 255],
                        [0,   0,   0]]
            }
        ]
        
        g = np.zeros(image.shape)
        o = np.empty(image.shape)
        fo = np.zeros(image.shape)
        gx = np.zeros(image.shape)
        gy = np.zeros(image.shape)
        
        # Find Edges Vertical & Horizontal
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                ix = 0
                iy = 0
                for x1 in range(max(x - 1, 0), min(x + 1, image.shape[0])):
                    for y1 in range(max(y - 1, 0), min(y + 1, image.shape[1])):
                        sg = sgx[x1 - x][y1 - y]
                        ix += int(image[x1, y1]) * sg
                for x1 in range(max(x - 1, 0), min(x + 1, image.shape[0])):
                    for y1 in range(max(y - 1, 0), min(y + 1, image.shape[1])):
                        sg = sgy[x1 - x][y1 - y]
                        iy += int(image[x1, y1]) * sg
                gx[x, y] = ix
                gy[x, y] = iy
                
        # Combine Edges
        gmax = 0
        for x in range(gx.shape[0]):
            for y in range(gx.shape[1]):
                gn = math.floor(math.sqrt(gx[x, y] ** 2 + gy[x, y] ** 2))
                if gn > gmax: gmax = gn
                g[x, y] = gn
        
        
        # Normalize + Calculate Direction
        thresold_avg = 0  
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                gm = g[x, y] / gmax * 255 # Gradient Magnitude
                go = math.degrees(math.atan2(gy[x, y], gx[x, y])) # Gradient Orientation
                if go < 0: go += 180
                g[x, y] = gm
                thresold_avg += g[x, y]
                o[x, y] = go

        thresold_avg = thresold_avg / (image.shape[0] * image.shape[1]) * 1.5
        print(f"Threshold Average: {thresold_avg}")
        
        # Apply thresold
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if g[x, y] < thresold_avg:
                    g[x, y] = 0
                    o[x, y] = None
                # else:
                #     g[x, y] = 255
                    
        # Thin edges
        for x in range(o.shape[0] - 2):
            dx = x + 1
            for y in range(o.shape[1] - 2):
                dy = y + 1
                theta = o[dx, dy]
                q = 0
                r = 0
                if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                    q = g[dx, dy+1]
                    r = g[dx, dy-1]
                elif (22.5 <= theta < 67.5):
                    q = g[dx+1, dy-1]
                    r = g[dx-1, dy+1]
                elif (67.5 <= theta < 112.5):
                    q = g[dx+1, dy]
                    r = g[dx-1, dy]
                elif (112.5 <= theta < 157.5):
                    q = g[dx-1, dy-1]
                    r = g[dx+1, dy+1]
                
                if (g[dx, dy] >= q) and (g[dx, dy] >= r):
                    fo[dx, dy] = g[dx, dy]
                    # fo[dx, dy] = 255
                else:
                    fo[dx, dy] = 0
        
        # aradius = 1
        # adiam = (aradius * 2) + 1
        # for x in range(o.shape[0] // adiam):
        #     dx = x * adiam
        #     for y in range(o.shape[1] // adiam):
        #         dy = y * adiam
        #         avg = 0
        #         amt = 0
                
        #         # Get Average direction
        #         for ox in range(adiam):
        #             for oy in range(adiam):
        #                 if o[dx + ox, dy + oy] == None:
        #                     continue
        #                 avg += o[dx + ox, dy + oy]
        #                 amt += 1
        #         map = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                
        #         # Calculate Single Line
        #         if (amt != 0):
        #             avg = avg / amt
        #             if avg > 180: print(f"{avg} Greater than 180")
        #             for a in amap:
        #                 if (a["Min"] <= avg and avg <= a["Max"]):
        #                     map = a["Map"]
        #                     break
        #         for ox in range(adiam):
        #             for oy in range(adiam):
        #                 fo[dx + ox, dy + oy] = map[ox][oy]
                
        # return g.astype(np.uint8)
        # return fo.astype(np.uint8)
        return (fo.astype(np.uint8), g.astype(np.uint8))
    
    
    def GetG(image: ndarray):
        sgx = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        sgy = [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        
        g = np.zeros(image.shape)
        gx = np.zeros(image.shape)
        gy = np.zeros(image.shape)
        
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                ix = 0
                iy = 0
                for x1 in range(max(x - 1, 0), min(x + 1, image.shape[0])):
                    for y1 in range(max(y - 1, 0), min(y + 1, image.shape[1])):
                        sg = sgx[x1 - x][y1 - y]
                        ix += int(image[x1, y1]) * sg
                for x1 in range(max(x - 1, 0), min(x + 1, image.shape[0])):
                    for y1 in range(max(y - 1, 0), min(y + 1, image.shape[1])):
                        sg = sgy[x1 - x][y1 - y]
                        iy += int(image[x1, y1]) * sg
                gx[x, y] = ix
                gy[x, y] = iy
                
        # Combine Edges
        gmax = 0
        for x in range(gx.shape[0]):
            for y in range(gx.shape[1]):
                gn = math.floor(math.sqrt(gx[x, y] ** 2 + gy[x, y] ** 2))
                if gn > gmax: gmax = gn
                g[x, y] = gn
        
        # Normalize + Calculate Direction
        thresold_avg = 0  
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                gm = g[x, y] / gmax * 255 # Gradient Magnitude
                g[x, y] = gm
                thresold_avg += g[x, y]

        thresold_avg = thresold_avg / (image.shape[0] * image.shape[1]) * 2
        print(f"Threshold Average: {thresold_avg}")
        
        for x in range(g.shape[0]):
            for y in range(g.shape[1]):
                if g[x, y] < 50:
                    g[x, y] = 0
                
                
        return g.astype(np.uint8)
    
    def GetGuassian(x: int, y: int, sigma=1):
        return math.pow(math.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * (sigma ** 2))
        # return (1 / (2 * math.pi * (sigma ** 2))) * math.pow(math.e, (-((x * x + y * y) / (2 * (sigma * sigma)))))
    