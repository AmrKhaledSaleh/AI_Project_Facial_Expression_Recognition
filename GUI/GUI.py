import tkinter as tk
from tkinter import filedialog


class GUI:
    def __init__(self):
        # Create the window and set properties
        self.root = tk.Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.title("Face Expression Detector")

        # Create app Frames
        self.MainFrame = tk.Frame(self.root)
        self.ModelFrame = tk.Frame(self.root)
        self.PhotoFrame = tk.Frame(self.root)
        self.VideoFrame = tk.Frame(self.root)
        self.CameraFrame = tk.Frame(self.root)

        # Home Button to get back to MainFrame
        self.home = tk.Button(self.root, text="Home", font=('Arial', 18),command=self.Home)
        self.home.place(x=10, y=10, height=40, width=80)

        # model Variable contain the model as string "ANN" or "CNN"
        self.model = ""

        # Call the main frame to start the app
        self.Make_ModelFrame()

        # load the app
        self.root.mainloop()

    def set_model_cnn(self):
        self.model = "CNN"
        self.ModelFrame.pack_forget()
        self.Make_MainFrame()
        print("Selected model: CNN")

    def set_model_ann(self):
        self.model = "ANN"
        self.ModelFrame.pack_forget()
        self.Make_MainFrame()
        print("Selected model: ANN")

    def Home(self):
        self.ModelFrame.pack()
        self.MainFrame.pack_forget()
        self.PhotoFrame.pack_forget()
        self.VideoFrame.pack_forget()
        self.CameraFrame.pack_forget()

    def Make_MainFrame(self):
        for i in range(4):
            self.MainFrame.grid_rowconfigure(i, weight=1)  # Allow rows to expand vertically
        self.MainFrame.grid_columnconfigure(0, weight=1)  # Allow column to expand horizontally

        # the label on the top
        self.Mainlabel = tk.Label(self.MainFrame, text="Choose Input Type", font=('Arial', 36))

        '''
            - each button when you click it , Calls the new frame and make the MainFrame invisible
            - each frame call the Make_ModelFrame that is responsible for chossing the model
        '''
        self.PhotoButton = tk.Button(self.MainFrame, text="Photo", font=('Arial', 24), command=self.Make_PhotoFrame)
        self.VideoButton = tk.Button(self.MainFrame, text="Video", font=('Arial', 24), command=self.Make_VideoFrame)
        self.CameraButton = tk.Button(self.MainFrame, text="Camera", font=('Arial', 24), command=self.Make_CameraFrame)

        self.Mainlabel.grid(row=0, column=0, padx=30, pady=80, sticky="nsew")
        self.PhotoButton.grid(row=1, column=0, padx=30, pady=30, sticky="nsew")
        self.VideoButton.grid(row=2, column=0, padx=30, pady=30, sticky="nsew")
        self.CameraButton.grid(row=3, column=0, padx=30, pady=30, sticky="nsew")

        self.MainFrame.pack()

    def Make_ModelFrame(self):
        for i in range(3):
            self.ModelFrame.grid_rowconfigure(i, weight=1)
        self.ModelFrame.grid_columnconfigure(0, weight=1)

        self.Modellabel = tk.Label(self.ModelFrame, text="Choose Model", font=('Arial', 36))

        self.ModelANN = tk.Button(self.ModelFrame, text="ANN", font=('Arial', 24), command=self.set_model_ann)
        self.ModelCNN = tk.Button(self.ModelFrame, text="CNN", font=('Arial', 24), command=self.set_model_cnn)

        self.Modellabel.grid(row=0, column=0, padx=30, pady=100, sticky="nsew")
        self.ModelANN.grid(row=1, column=0, padx=30, pady=30, sticky="nsew")
        self.ModelCNN.grid(row=2, column=0, padx=30, pady=30, sticky="nsew")

        self.ModelFrame.pack()

    def open_file_explorer(self):

        file_path = filedialog.askopenfilename(title="Select Photo",
                                               filetypes=[("Image Files", "*.jpg *.jpeg *.png")], )

        # Update the photo path entry with the selected file path
        self.photo_path_entry.configure(state="normal")
        self.photo_path_entry.delete(0, tk.END)
        self.photo_path_entry.insert(0, file_path)
        self.photo_path_entry.configure(state="readonly")

    def Make_PhotoFrame(self):
        self.MainFrame.pack_forget()

        for i in range(3):
            self.PhotoFrame.grid_rowconfigure(i, weight=1)
        self.PhotoFrame.grid_columnconfigure(0, weight=1)

        # Button for browsing photo
        self.browse_photo_button = tk.Button(self.PhotoFrame, text="Browse Photo", font=("Arial", 20),command=self.open_file_explorer)
        self.browse_photo_button.grid(row=0, column=0, pady=20)

        # Label to display selected photo path
        self.photo_path_label = tk.Label(self.PhotoFrame, text="Selected Photo Path:", font=("Arial", 16))
        self.photo_path_label.grid(row=1, column=0, pady=10)

        # Placeholder for displaying the selected photo path
        self.photo_path_entry = tk.Entry(self.PhotoFrame, font=("Arial", 16), state="normal")
        self.photo_path_entry.grid(row=2, column=0, pady=10)

        self.PhotoFrame.pack()

    def Make_VideoFrame(self):
        self.MainFrame.pack_forget()

        # hey , model variable has the selected model as a string
        # Write your code here bro

        self.VideoFrame.pack()

    def Make_CameraFrame(self):
        self.MainFrame.pack_forget()

        # hey , model variable has the selected model as a string
        # Write your code here bro

        self.CameraFrame.pack()


GUI()
