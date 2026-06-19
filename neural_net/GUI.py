import tkinter as tk
import numpy as np
from scipy.ndimage import center_of_mass,shift 
from PIL import Image, ImageDraw, ImageTk

from network import Network
from layers import Linear, ReLU, SoftMax
from MNIST_test import load_model

# config for GUI layout and drawing
NET_W, NET_H = 600, 600
MID_SIZE = 200
DRAW_SIZE = 280
MAX_NODES = 12
BG_COLOR = 'White'
EDGE_COLOR = 'Gray'
NODE_COLOR = 'Gray'
NODE_ACTIVE_COLOR = 'Black'
NODE_ACTIVE_TEXT_COLOR = 'White'
NODE_TEXT_COLOR = 'Black'
EDGE_ACTIVE_COLOR = 'Black'

...

""" 
This create a GUI where user can draw a digit and see the network's prediction. It also visualizes the network structure and activations.
Use tinker for GUI,
Structure:
    TK = main window
    ├── Frame = main container
        ├── Left Frame = drawing board
        │   ├── Canvas = where user draws
        │   ├── Buttons = Predict, Clear
        │   └── Label = shows prediction
        └── Right Frame = network visualization
            └── Canvas = where we draw the network structure and activations

"""
# Pack() is put compunts on main window

class Visualizer:
    def __init__(self, root, model):
        self.model = model
        root.title("Simple Neural Network Visualizer")

        frame = tk.Frame(root); 
        frame.pack()

        """ Drawing board on the left for user to draw a digit use moues and use canvas to get the drawing through google search """
        left = tk.Frame(frame); 
        left.pack(side=tk.LEFT)
        self.draw_canvas = tk.Canvas(left, width=DRAW_SIZE, height=DRAW_SIZE, bg="black", cursor="cross")
        self.draw_canvas.pack()
        # drag motion of mouse with left button held
        self.last_xy = None
        self.draw_canvas.bind("<B1-Motion>", self.paint)
        self.draw_canvas.bind("<ButtonRelease-1>", self.release)
        
        """ Use PIL to convert take of output of canvas and convert it to an image in pixel array that the model can use. """
        self.image = Image.new("L", (DRAW_SIZE, DRAW_SIZE), 'BLACK')
        self.draw = ImageDraw.Draw(self.image)
        """ Buttons for user to interact with the application """
        btns = tk.Frame(left); 
        btns.pack(fill=tk.X, pady=5)
        # command is equal to java's event listener, when button is clicked it calls the function predict or clear
        tk.Button(btns, text="Predict", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btns, text="Clear", command=self.clear).pack(side=tk.LEFT)
        # text for the window 
        self.label = tk.Label(left, text="Draw a digit", font=("Arial", 16))
        self.label.pack()

        middle = tk.Frame(frame)
        middle.pack(side=tk.LEFT, padx=10)
        tk.Label(middle, text="Network Sees", font=("Arial", 12)).pack()
        self.seen_canvas = tk.Canvas(middle, width=MID_SIZE, height=MID_SIZE, bg="black")
        self.seen_canvas.pack()

        # View the network nodes, edges 
        # Connect to right with frame and canvas with frame and make the show on the frame
        #
        self.net_canvas = tk.Canvas(frame, width=NET_W, height=NET_H, bg=BG_COLOR)  
        self.net_canvas.pack(side=tk.LEFT, padx=10, pady=10)

        
        self.positions = self.node_positions(self.layer_sizes())
        self.draw_network()

    #  Drawing board 
    """ e is event that user is drawing on the canvas, which the mouse cursor x and y position that use drag """
    def paint(self, e):
        r = 6
        if self.last_xy is not None:
            x_prev, y_prev = self.last_xy
            self.draw_canvas.create_line(x_prev, y_prev, e.x, e.y,
                                         fill="white", width=r*2,
                                         capstyle=tk.ROUND, smooth=True)
            self.draw.line([x_prev, y_prev, e.x, e.y], fill=255, width=r*2)
        self.draw_canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r,
                                     fill="white", outline="white")
        self.draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=255)
        self.last_xy = (e.x, e.y)

    def release(self, e):
        self.last_xy = None
        self.predict()   # live prediction on mouse-up!

    def clear(self):
        self.draw_canvas.delete("all")
        self.draw.rectangle([0, 0, DRAW_SIZE, DRAW_SIZE], fill=0)
        self.label.config(text="Draw a digit")
        self.seen_canvas.delete("all")
        self.draw_network()

    def preprocess(self):
        img = np.asarray(self.image)
        if img.max() == 0 :
            return None
        rows = np.any(img > 0, axis = 1)
        cols = np.any(img > 0, axis = 0)
        r_min, r_max = np.where(rows)[0][[0,-1]]
        c_min, c_max = np.where(cols)[0][[0,-1]]
        digit = img[r_min: r_max+1, c_min: c_max+1]
        # bounding box 

        """ Finish it tomorow 
         resize boundbox to 20 x 20 
            Place it within 28 x 28 
        and centered the normalize it /255.0

        """

        height, width = digit.shape
        
        if height > width:
            Newheight = 20
            NewWidth = max(1, round(( width*20) / height))
        else:
            NewWidth = 20
            Newheight = max(1, round((height *20) /width ))
        # scale factor
        """
        Method is 
        if h > w 
            h be map to 20 
            some factor h * f = 20
            f = 20 / h
            apply f onto w
            w * (20 /h )
            using max to elimate zero
        """
        pil_digit = Image.fromarray(digit.astype(np.uint8))
        pil_digit = pil_digit.resize((NewWidth, Newheight),Image.LANCZOS)
        pil_digit = np.array(pil_digit)
        grid = np.zeros((28,28))
        startx = (28- NewWidth) // 2
        starty = (28 - Newheight) //2
        # center it 
        grid[starty:starty+Newheight, startx:startx + NewWidth] = pil_digit
        # copy paste it on the 28 x 28 grid in the center 
        cy, cx = center_of_mass(grid)
        if not (np.isnan(cy) or np.isnan(cx)):
            shifty, shiftx = 14 - cy, 14 - cx
            grid = shift(grid, [shifty, shiftx], mode='constant', cval=0)
        """ most width is, make width in the "perfect" center  """
        grid = grid.reshape(784,1)
        grid = grid / 255.0
        return grid

        ...



#  Network structure 
    def layer_sizes(self):
        size = []
        flag = True
        for layer in self.model.layers:
            temp = [2] 
           
            if isinstance(layer, Linear):
                # print(layer.W.shape)
                if flag :
                    size.append(layer.W.shape[1])
                    flag = False 
                size.append(layer.W.shape[0])
                
        return size
        # read [784,128,64,10] from Linear W shapes
        # (W is (in, out): shape[0]=in for first layer, then shape[1]=out)
        ...


# How to get the node positions
# 
    def node_positions(self,sizes):
        nodes = []
        """
        temp = [2] 
        temp[0] = layer.w.shape[1]
        temp[1] = layer.w.shape[0]
        node.append(temp)

        """

        numLayers = len(sizes)
        for idx,size in enumerate(sizes):
            visible_node = min(MAX_NODES,size)
            x = int(NET_W * ((idx + 1) /( numLayers+1)) )
            # Oli fourmala 
            temp = []
            for yidx in range(visible_node):
                y = int(NET_H * ((yidx + 1) /( visible_node+1)) )
                temp.append([x, y])
            nodes.append(temp)
        return nodes
                # get the position of each node in the layer and store it in a list of lists
                # return list of lists of (x,y) positions for each node in each layer
        # (x,y) for up to MAX_NODES per layer; return list of lists
        ...
    """
    record each forward outputs for each layers 
     """
    def forward_capture(self, x):
        activations = [x]
        out = x
        for layer in self.model.layers:
            out = layer.forward(out)
            if isinstance(layer, (Linear, SoftMax)):
                activations.append(out) 
        return activations, out
        ...

    def act_color(self, value, vmax):
        # TODO: map activation 0..vmax to a color (dark -> bright)
        ...

    def draw_network(self, activations=None):
        self.net_canvas.delete("all")
        # TODO: draw edges
        # TODO: draw nodes (color by activation if provided, else default)
        # TODO: draw size labels under each layer
        ...

    #Predict 
    def predict(self):
        x = self.preprocess()
        if x is None:
            self.label.config(text="Draw something first!")
            return
        # Show the networks img
        self.show_midimage(x)
        activations, out = self.forward_capture(x)
        digit = int(np.argmax(out, axis=0)[0])
        self.label.config(text=f"Prediction: {digit}")
        # self.draw_network(activations)

    def show_midimage(self, preprocessed):
        """Display the 28x28 preprocessed image scaled up."""
        if preprocessed is None:
            self.seen_canvas.delete("all")
            return
        # Reshape (784,1) back to (28,28) and scale to [0,255]
        img = (preprocessed.reshape(28,28)*255).astype(np.uint8)

        # Convert to PIL image then upscale
        pil = Image.fromarray(img, mode='L')
        pil = pil.resize((MID_SIZE,MID_SIZE), Image.NEAREST)

        # Convert to Tkinter image
        self.mid_tk_image = ImageTk.PhotoImage(pil)

        # Display
        self.seen_canvas.delete("all")
        self.seen_canvas.create_image(0,0, anchor=tk.NW, image=self.mid_tk_image)


if __name__ == "__main__":

    model = load_model('mnist_weights.npz')

    root = tk.Tk()
    Visualizer(root, model)
    root.mainloop()