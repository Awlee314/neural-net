import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw

from network import Network
from layers import Linear, ReLU, SoftMax
from MNIST_test import load_model

# config for GUI layout and drawing
NET_W, NET_H = 600, 600
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

        self.draw_canvas.bind("<B1-Motion>", self.paint)
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

        # View the network nodes, edges 
        # Connect to right with frame and canvas with frame and make the show on the frame
        #
        self.net_canvas = tk.Canvas(frame, width=NET_W, height=NET_H, bg=BG_COLOR)  
        self.net_canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.sizes = self.layer_sizes()
        self.positions = self.node_positions(self.sizes)
        self.draw_network()

    #  Drawing board 
    """ e is event that user is drawing on the canvas, then  """
    def paint(self, e):
        r = 10
        self.draw_canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r,
                                     fill="white", outline="white")
        self.draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=255)

    def clear(self):
        self.draw_canvas.delete("all")
        self.draw.rectangle([0, 0, DRAW_SIZE, DRAW_SIZE], fill=0)
        self.label.config(text="Draw a digit")
        self.draw_network()

    def preprocess(self):
        # TODO: crop bbox, scale to 20x20, center in 28x28,
        # normalize /255, return (784,1) — or None if blank
        ...



#  Network structure 
    def layer_sizes(self):
        size = []
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                size.append(layer.W)
        return size
        # read [784,128,64,10] from Linear W shapes
        # (W is (in, out): shape[0]=in for first layer, then shape[1]=out)
        ...


# How to get the node positions
# 
    def node_positions(self, sizes):
        nodes = []
        for x in range(sizes.W.shape[0]):
            for y in range(sizes.W.shape[1]):
                nodes.append((x,y))
        return nodes
                # get the position of each node in the layer and store it in a list of lists
                # return list of lists of (x,y) positions for each node in each layer
        # (x,y) for up to MAX_NODES per layer; return list of lists
        ...

    def forward_capture(self, x):
        # TODO: run model layer by layer, collect output after each
        # Linear/SoftMax layer. return (activations_list, final_output)
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
        activations, out = self.forward_capture(x)
        digit = int(np.argmax(out, axis=0)[0])
        self.label.config(text=f"Prediction: {digit}")
        self.draw_network(activations)


if __name__ == "__main__":
    model = Network([Linear(784,128), ReLU(),
                     Linear(128,64), ReLU(),
                     Linear(64,10), SoftMax()])
    model = load_model('mnist_weights.npz')

    root = tk.Tk()
    Visualizer(root, model)
    root.mainloop()