from tkinter import *
import tkinter as tk
from tkinter import filedialog
import os
from FRI_TL import train_main
from classify import classify_main

#https://stackoverflow.com/questions/34276663/tkinter-gui-layout-using-frames-and-grid

folder_path = "adsdf"
folder_counter = 0
entry_name = "none"

epoch = 0
batch_size = 0
num_class = 0

'''
def print_values():
    print(folder_path)
    print(folder_counter)
    print(entry_name)

print("old values")
print_values()
'''

global selected_model_path
global trained_text_file    # Variable used for getting path of txt file from model folder
model_dir_path = r"D:/Dev_Con_Project/wri_devcon_object_detection/model/"

def train():
    used_to_train = []  # An array of all of the animals used to train a model
    def browse_button():
    # Allow user to select a directory and store it in global variable called folder_path
        global folder_path
        global folder_counter
        global entry_name

        entry_name = model_name.get()   # Gets name from entry field of GUI
        folder_path = filedialog.askdirectory(title='Select your pictures folder')
        folder_counter = 0
        files = os.listdir(folder_path)

        for fn in files:
            if not fn.startswith('.') and fn != 'Thumbs.db':    # Ignores hidden files when counting num of folders
                used_to_train.append(fn)
                folder_counter+=1

        print("These animals are used to train this model: %s" % used_to_train)
        num_class = folder_counter
        root.attributes('-topmost', True)   # Keeps GUI window at the front of your screen

        Label(btm_frame, text=folder_path, bg='#8B9DC3').grid(row=4, sticky=W, padx=100)
        # print("updated values")
        # print_values()
    
        
    root = Toplevel()
    root.title('Model Definition')
    root.geometry('{}x{}'.format(706, 500))


    # create all of the main containers
    top_frame = Frame(root, width=700, height=100, pady=3)
    top_frame.config(background='#8B9DC3')
    center = Frame(root, bg='gray2', width=700, height=300, padx=3, pady=3)
    center.config(background='#DFE3EE')
    btm_frame = Frame(root, width=700, height=150, pady=3)
    btm_frame.config(background='#8B9DC3')

    # layout all of the main containers
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)


    top_frame.grid(row=0, sticky="ew")
    center.grid(row=1, sticky="nsew")
    btm_frame.grid(row=3, sticky="ew")





    # Create the widgets for the top frame
    browse_label = Label(top_frame, text='Please select the folder used for the training of the model:', bg='#8B9DC3').grid(row=0, column=0, sticky=W, padx=(100,0))
    browse_btn = Button(top_frame, text="Browse", command=browse_button).grid(row=0, column=1, sticky=W, padx=(10,0))

    # Gets the name entered in the text field and makes a folder to store the model
    def get_model_name():
        entry_name = model_name.get()  # Gets name from entry field of GUI
        print("Model name: %s" % entry_name)

        if os.path.isdir(model_dir_path) and not(os.path.isdir(model_dir_path + '/' + entry_name)): # Models folder exists and unique model
            os.mkdir(model_dir_path + '/' + entry_name)

        elif os.path.isdir(model_dir_path) and os.path.isdir(model_dir_path + '/' + entry_name):  # Models folder and model already exist
            Label(top_frame, text='Model name already exists!', borderwidth=1, bg='#8B9DC3').grid(row=2, column=0, sticky=W, padx=(100, 0))

        else:   # Models folder doesnt exist, create it and the new model subfolder
            os.mkdir(model_dir_path)
            os.mkdir(model_dir_path + '/' + entry_name)

    """ Currently not working trying to make warning display in constant time as we type in a model's name to be made
    array_entry = []
    def click(key):
        array_entry.append(key.char)
        x = ''.join([str(i) for i in array_entry])
        print(x)
        warning2 = Label(top_frame, text='Model name already exists!', borderwidth=1, bg='red')
        warning2.grid(row=2, column=0, sticky=W, padx=(100, 0))
        warning2.grid_forget()
        print(warning2.winfo_exists() and not (os.path.isdir(path + '/' + str(x))))
        if os.path.isdir(path + '/' + str(x)):
            warning2.grid(row=2, column=0, sticky=W, padx=(100, 0))

        if(warning2.winfo_exists() and not(os.path.isdir(path + '/' + str(x)))):
            warning2.destroy()
    """

    # Layout the widgets in the top frame
    model_label = Label(top_frame, text='Please suggest the name of the model:', borderwidth=1, bg='#8B9DC3').grid(row=1, column=0, sticky=W, padx=(100, 0))
    model_name = Entry(top_frame, width=15)
    model_name.grid(row=1, column=1, sticky=W, padx=(10,0))
    # model_name.bind("<Key>", click)
    # button2.grid(row=0, column = 1, padx=(100, 10))


    # Create the center widgets
    center.grid_rowconfigure(0, weight=1)
    center.grid_columnconfigure(1, weight=1)

    text1 = ("Please select the Folder used for the training of the model:\n")*10

    ctr_left = Frame(center, width=100, height=190)
    ctr_left.config(background='#DFE3EE')
    ctr_mid = Frame(center, width=500, height=100)
    ctr_mid.config(background='#DFE3EE')
    ctr_right = Frame(center, width=100, height=190)
    ctr_right.config(background='#DFE3EE')

    # https: // www.python - course.eu / tkinter_text_widget.php

    S = Scrollbar(ctr_mid)
    T = Text(ctr_mid, height=100, width=500)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=RIGHT, fill=Y)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)

    quote = """TRAINING INSTRUCTIONS:
    
    PRECONDITIONS: 
    
    - The folder selected above should contain subfolders
      for all of the animals that you intend to train 
      this model on. Animals belonging to one species 
      should not be mixed in the wrong species folder. 
      This can cause inaccurate training of the model.
      
    - The naming of the animal species subfolders should 
      not contain any spaces, underscores may be used if 
      separation is required.
      
    - The name of the model entered in the field above must
      be vaild and unique. Once training has concluded, the
      trained model will be saved in the model folder and 
      will possess the name entered in the above field by 
      the user.
    """

    T.insert(END, quote)


    ctr_left.grid(row=0, column=0, sticky="ns")
    ctr_mid.grid(row=0, column=1, sticky="nsew")
    ctr_right.grid(row=0, column=2, sticky="ns")

    def start_training():
        entry_name = model_name.get()
        root.destroy()
        train_main(folder_counter, folder_path, entry_name, model_dir_path, used_to_train)


    start_with_toggle = Button(btm_frame, text='Start Training', state=DISABLED, command = lambda:[get_model_name(), start_training()])
    start_with_toggle.grid(row=2, sticky=W, padx=500, pady=10)
    
    var = BooleanVar()

    # Toggles start button from disabled to enabled
    def toggle():
        
        if var.get() and not(model_name.get() == "none") and not(folder_counter == 0):
            start_with_toggle['state'] = 'normal'

        else:
            start_with_toggle['state'] = 'disable'
            warning = Label(btm_frame, text="Please select a populated folder to train and provide a name for the model. Then re-check the box!", bg='#8B9DC3', fg="red")
            warning.grid(row=3, sticky=W, padx=100, pady=10)
    
    check_btn = Checkbutton(btm_frame, text="I have read all the instructions needed for the training", variable=var, command=toggle, bg='#8B9DC3').grid(row=2, sticky=W,  padx=100)

    instruction_label = Label(btm_frame, text="Please check the following box:", bg='#8B9DC3').grid(row=1, sticky=W, padx=100)

    root.mainloop()
   
def classify_gui():
    classifyWindow = Toplevel()
    classifyWindow.geometry('{}x{}'.format(706, 500))

    top_frame_classify = Frame(classifyWindow, width=700, height=100, pady=3)
    center_classify = Frame(classifyWindow, width=700, height=300, padx=3, pady=3)
    btm_frame_classify = Frame(classifyWindow, width=700, height=150, pady=3)

    classifyWindow.grid_rowconfigure(1, weight=1)
    classifyWindow.grid_columnconfigure(0, weight=1)

    top_frame_classify.grid(row=0, sticky="ew")
    top_frame_classify.config(background='#8B9DC3')
    center_classify.grid(row=1, sticky="nsew")
    center_classify.config(background='#DFE3EE')
    btm_frame_classify.grid(row=3, sticky="ew")
    btm_frame_classify.config(background='#8B9DC3')

    # create the center widgets
    center_classify.grid_rowconfigure(0, weight=1)
    center_classify.grid_columnconfigure(1, weight=1)


    ctr_left_classify = Frame(center_classify, bg='blue', width=100, height=190)
    ctr_mid_classify = Frame(center_classify, bg='yellow', width=500, height=100)
    ctr_right_classify = Frame(center_classify, bg='green', width=100, height=190)

    S = Scrollbar(ctr_mid_classify)
    T = Text(ctr_mid_classify, height=100, width=500)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=RIGHT, fill=Y)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)

    quote = """CLASSIFY INSTRUCTIONS:

    PRECONDITIONS: 
    
    - In the above drop-down menu you must select the
      model you wish to use to classify images.
      
    - A folder containing the images you want to 
      classify must be selected. Once the 
      classification process is complete a folder with
      the name format of the current YYYY/MM/DD/TIME 
      will be created and animal species subfolders 
      will be created within it. Animal images will 
      have been classified by putting them in the 
      appropriately named subfolder which matches the 
      animal captured in the image.
    """

    T.insert(END, quote)

    ctr_left_classify.grid(row=0, column=0, sticky="ns")
    ctr_left_classify.config(background='#DFE3EE')
    ctr_mid_classify.grid(row=0, column=1, sticky="nsew")
    ctr_mid_classify.config(background='#DFE3EE')
    ctr_right_classify.grid(row=0, column=2, sticky="ns")
    ctr_right_classify.config(background='#DFE3EE')

    def select_button():
    # Allow user to select a directory and store it in global variable called folder_path_classify
        global folder_path_classify
        global folder_counter_classify

        folder_path_classify = filedialog.askdirectory(title='Select your pictures folder to classify')
        folder_counter_classify = 0
        files_classify = os.listdir(folder_path_classify)

        for fn in files_classify:   # Counts num of selected folders, ignoring hidden files/folders
            if not fn.startswith('.') and fn != 'Thumbs.db':
                folder_counter_classify+=1

        classifyWindow.attributes('-topmost', True)
        print(folder_counter_classify)
        print(folder_path_classify)

        # Label(btm_frame_classify, text=folder_path_classify, bg='#8B9DC3').grid(row=4, column=0, sticky=W, padx=100)
        Label(btm_frame_classify, text=folder_path_classify, bg='#8B9DC3').grid(row=2, column=0, sticky=W, padx=100)

    Label(top_frame_classify, text='Please select the folder used to classify:', bg='#8B9DC3').grid(row=0, column=0, padx=(100, 0))
    Button(top_frame_classify, text="Select", command=select_button).grid(row=1, column=1, sticky=W, padx=(10, 0))

    model_names = []
    model_paths = []
    text_file_paths = []
    # model_dir_path = r"D:/Dev_Con_Project/wri_devcon_object_detection/model/"

    for dir_, _, files in os.walk(model_dir_path):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, model_dir_path)
            rel_file = os.path.join(rel_dir, file_name).replace("\\", "/")
            rel_file = os.path.join(model_dir_path, rel_file).replace("\\", "/")
            str(rel_file)

            if file_name.endswith('.model'):
                model_names.append(file_name)   # Creates an array of all the models that exist
                model_paths.append(rel_file)    # Creates an array of all of the model paths

            if file_name.endswith('.npy'):
                text_file_paths.append(rel_file)    # Creates an array of all of the text file paths

    print(text_file_paths)
    print("available model paths: %s" % model_paths)

    variable = StringVar(classifyWindow)
    variable.set(model_names[0])  # Default value

    drop_down_btn = OptionMenu(top_frame_classify, variable, *model_names)
    drop_down_btn.grid(row=0, column=1, sticky=W, padx=(10, 0))
    drop_down_label = Label(top_frame_classify, text='Please select the trained model for classifying:', bg='#8B9DC3').grid(row=0, column=0, sticky=W, padx=(100, 0))


    def start_classify():
        print("value is:" + variable.get())
        for models in model_paths:
            path_seg = models.rpartition('/')
            if path_seg[2] == variable.get():
                selected_model_path = models
                print("selected model to use: %s" % path_seg[2])
                print("selected model path: %s" % selected_model_path)

        for text_files in text_file_paths:
            text_path_seg = text_files.rpartition('/')
            selected_name = variable.get()
            selected_name, _ = selected_name.split(".")   # Taking name of model selected, ignoring .model part
            if os.path.basename(os.path.dirname(text_files)) == selected_name:  # Compares base folder name & selected
                trained_text_file = text_files
                print("text file path: %s" % trained_text_file)

        classifyWindow.destroy()
        classify_main(selected_model_path, trained_text_file, folder_path_classify)

    classify_button = Button(btm_frame_classify, text="Start Classification", command=start_classify)
    # classify_button = Button(btm_frame_classify, text="Start Classification", state=DISABLED, command=start_classify)
    classify_button.grid(row=2, sticky=W, padx=500, pady=10)
    '''
    classify_var = BooleanVar()
    
    # Toggles start button from disabled to enabled
    def toggle_classify():

        if classify_var.get() and not (folder_counter_classify == 0):
            classify_button['state'] = 'normal'

        else:
            classify_button['state'] = 'disable'
            warning = Label(btm_frame_classify, text="Please select a populated folder to classify and select the desired model to use. Then re-check the box!", bg='#8B9DC3', fg="red")
            warning.grid(row=3, sticky=W, padx=100, pady=10)

    check_btn = Checkbutton(btm_frame_classify, text="I have read all the instructions needed for the classification", variable=classify_var, command=toggle_classify, bg='#8B9DC3').grid(row=2, sticky=W, padx=100)

    instruction_label = Label(btm_frame_classify, text="Please check the following box:", bg='#8B9DC3').grid(row=1, sticky=W, padx=100)
    '''
mWindow = tk.Tk()

# You can set any size you want
mWindow.geometry('600x500')
mWindow.title('Animal Classifier')
wtitle = tk.Label(mWindow, text="Please click the following option:", fg='blue')
wtitle.place(x=10, y=50)

# You can set any height and width you want
mmbutton = tk.Button(mWindow, height=5, width=20, text="Training", command=train)
mmbutton.place(x=100, y=140) 
classifybutton = tk.Button(mWindow, height=5, width=20, text="Classify", command=classify_gui)
classifybutton.place(x=300, y=140) 

mWindow.mainloop()
