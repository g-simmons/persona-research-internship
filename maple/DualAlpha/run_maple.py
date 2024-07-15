import os
import subprocess
import time
import sys
import shutil
import pandas as pd
from pathlib import Path




def get_ontology_column_names():
    fieldnames = []
    temp_string = ""
    for i in range(1, 1537):        #1536-d
        temp_string = f"embedding dim {i}"
        fieldnames.append(temp_string)

    fieldnames.append("function_value")
    return fieldnames

def get_df_from_csv(csv_path: str):
    columns = get_ontology_column_names()
    df = pd.read_csv(csv_path, names=columns, header=0)
    print(df)
    return df

def run_maple(dataframe: pd.DataFrame, output_directory: str):
    os.mkdir("temp_maple_directory")
    dataframe.to_csv(path_or_buf="temp_maple_directory/temp_csv.csv", index = False)

    with open("temp_maple_directory/temp_maple_script.mpl", mode='w') as maple_script:
        maple_script.write('''read("include"):
                           
pd := randpd(100);
A := ImportMatrix("temp_maple_directory/temp_csv.csv");
[m, n] := Dimension(A);
A := A[2.., ..];
A := Matrix(A,datatype=float[8]);
powervec := A[..,3];
A := A[..,..2];
A := Matrix(A,datatype=float[8]);
pd := pdiag(A, powervec, 3);

pd:-draw();

X,Phi := getalpha(pd:-A,pd:-pow,pd:-a1,3);
X,Phi := getalpha(pd:-A,pd:-pow,3,3);

# getbetti(x,0..2,3);

display([pd:-draw(),seq(point(Phi(sig),color=blue,symbol=solidcircle,symbolsize=5),sig=X[0]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[1]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[2])]);
plotname := "temp_maple_directory/temp_alpha_complex_plot.png";
MyPlot := pd:-draw();
plotsetup(png, plotoutput=plotname, plotoptions="width=1024,height=768");
Export(plotname, MyPlot);
plotsetup(default);
printf("Plot saved to %s\n", plotname);''')
    
    
    with open("temp_maple_directory/log.txt", mode='w') as logger:
        logger.write("Beginning of log")
    log_path = Path("temp_maple_directory/log.txt")

    testing = subprocess.Popen(["maple", "temp_maple_directory/temp_maple_script.mpl"], stdout=log_path.open(mode="w"))

    plot_made = False
    while not plot_made:
        time.sleep(1)
        if "Plot saved to" in log_path.read_text():
            print("Temporary plot was saved")
            plot_made = True

    output_success = True
    if output_success:
        os.mkdir(output_directory)
        final_plot_path = output_directory + "/alpha_complex_plot.png"
        shutil.copyfile('temp_maple_directory/temp_alpha_complex_plot.png', final_plot_path)


    time.sleep(2)
    os.remove("temp_maple_directory/temp_alpha_complex_plot.png")
    os.remove("temp_maple_directory/log.txt")
    os.remove("temp_maple_directory/temp_csv.csv")
    os.remove("temp_maple_directory/temp_maple_script.mpl")
    os.rmdir("temp_maple_directory")
    print("All Finished")


my_df = get_df_from_csv("data/aeo_terms.csv")
print(my_df)

run_maple(my_df, "finaltesting")


# testing = subprocess.Popen(["maple", "pdws1_modified_copy_ontology.mpl"])

# testing = subprocess.Popen(["maple", "pdws1_modified_copy_ontology.mpl", "-I", "data/aeo_terms.csv"])

# time.sleep(5)

# testing.terminate()
# os.rmdir("testing")

