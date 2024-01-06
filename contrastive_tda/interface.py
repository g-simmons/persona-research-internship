import tkinter as tk
import pandas as pd
from tkinter import filedialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 

root = tk.Tk()
root.title("Dimensional Analysis")

# Create a Pandas DataFrame to store and edit prompts
df = pd.DataFrame()

def open_file():
    """Open data file for dimensional analysis"""
    file_path = filedialog.askopenfilename(title="Select a file")
    if(file_path):
        print(f"Selected {file_path}")
        read_file(file=file_path)

def read_file(file):
    """Read .xlsx file and store content in Pandas dataframe"""
    try:
        with open(file, 'rb') as file:        
            global df    
            df = pd.read_excel(file)
            df["full_prompt"] = df.apply(
                lambda x: concat_components(
                    x["edited_acting"],
                    x["edited_direction"],
                    x["edited_cinematography"],
                ),
                axis=1,
            )
            # Make a Tk label display complete message
            read_label = tk.Label(text=f"Finished reading file")
            read_label.grid(row=1, column=1)
            return df
    except Exception as e:
        print("Error reading file:", str(e))

def concat_components(acting, direction, cinematography):
    return acting + " " + direction + " " + cinematography

def save_file():
    raise NotImplementedError
# Create a button that opens the file dialog when clicked
open_button = tk.Button(root, text="Open File", command=open_file)
open_button.grid(row=0, column=1)

#-----------------Embedding Method Selection-------------------------
# Embedding methods
embedding_method_list = ["Cloud","Local","OpenAI"]
# Store selected emebedding method
select_emb = tk.StringVar(root)

# Create a label for embedding method
emb_method_label = tk.Label(text="Embedding")
emb_method_label.grid(row=2, column=0)

# Set the initial option
select_emb.set(embedding_method_list[0])

# Create the OptionMenu widget for embedding method
emb_dropdown_menu = tk.OptionMenu(root, select_emb, *embedding_method_list)
emb_dropdown_menu.grid(row=3, column=0)

#-----------------Dimensional Reduction Method-------------------------
# Dimensional reduction methods
dim_red_method = ["PCA", "t-SNE"]

# Store selected dimensional reduction method
select_dim_red = tk.StringVar(root)

# Create a label for dimensional reduction method
dim_red_label = tk.Label(text="Dimensional Reduction")
dim_red_label.grid(row=4, column=0)

# Set the intitial option
select_dim_red.set(dim_red_method[0])

# Create the OptionMenu widget for dimensional reduction method
dim_dropdown_menu = tk.OptionMenu(root, select_dim_red, *dim_red_method)
dim_dropdown_menu.grid(row=5, column=0)

#-----------------Sentiment Analysis Method-----------------------------
# Sentiment Analysis method
sentiment_method = ["Vader", "Text Blob"]

# Store selected sentiment analysis method
select_sentiment = tk.StringVar(root)

# Create a label for selected sentiment analysis method
sentiment_label = tk.Label(text="Sentimental Analysis")
sentiment_label.grid(row=6, column=0)

# Set the intitial option
select_sentiment.set(sentiment_method[0])

# Create the OptionMenu widget for sentiment analysis method
sentiment_dropdown_menu = tk.OptionMenu(root, select_sentiment, *sentiment_method)
sentiment_dropdown_menu.grid(row=7, column=0)

def embedding_data(select_emb):
    embedding_method = select_emb
    # Full prompt embedding
    for row in range(1,100):
        inp = [df.loc[row,"full_prompt"]]
        response = embedding_method(inp)
        df.loc[row,"full_prompt_embedding"] = str(response)
    # Edited Direction Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_direction"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        # if out of the edit window of directoin embeddings (1-33) use the original
        if row > 33:
            df.loc[row,"edited_direction_embedding"] = original_embedding
        else:
            inp = [df.loc[row,"edited_direction"]]
            response = embedding_method(inp)
            df.loc[row,"edited_direction_embedding"] = str(response)
    # Edited Action Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_acting"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 33 and row < 67:
            inp = [df.loc[row,"edited_acting"]]
            response = embedding_method(inp)
            df.loc[row,"edited_acting_embedding"] = str(response)
        else:
            df.loc[row,"edited_acting_embedding"] = original_embedding
    # Edited Cinematography Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_cinematography"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 66 and row < 100:
            inp = [df.loc[row,"edited_cinematography"]]
            response = embedding_method(inp)
            df.loc[row,"edited_cinematography_embedding"] = str(response)
        else:
            df.loc[row,"edited_cinematography_embedding"] = original_embedding
    # Store as "FINALOUTPUT + embedding_method.xlsx"
    df.to_excel(f"FINALOUTPUT_{embedding_method.__name__}.xlsx")
    return df

# Create a Button for embedding method
emb_button = tk.Button(root, text="Embed Data", command=lambda: embedding_data(select_emb))
emb_button.grid(row=3, column=1)

# Create a button for saving file
save_button = tk.Button(root, text="Save as file", command=save_file)
save_button.grid(row=9, column=0)

def plot(): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), dpi = 100) 
  
    # list of squares 
    y = [i**2 for i in range(101)] 
  
    # adding the subplot 
    plot1 = fig.add_subplot(111) 
  
    # plotting the graph 
    plot1.plot(y) 
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, root)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(row=3, column =4)
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, root) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().grid(row=4, column=4)

# button that displays the plot 
plot_button = tk.Button(root, command = plot, text = "Plot") 
  
# place the button  
# in main window 
plot_button.grid(row=8, column=0)

# Run the mainloop
root.mainloop()