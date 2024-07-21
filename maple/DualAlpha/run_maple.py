#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import shutil
import pandas as pd
from pathlib import Path
import tempfile
import ast


maple_script_str = '''read("include"):
                           
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

printf("BETTI_NUMBER_MARKER");
betti_numbers := getbetti(X,0..2,3);



display([pd:-draw(),seq(point(Phi(sig),color=blue,symbol=solidcircle,symbolsize=5),sig=X[0]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[1]),seq(point(Phi(sig),color=red,symbol=solidcircle,symbolsize=5),sig=X[2])]);
plotname := "temp_maple_directory/temp_alpha_complex_plot.png";
MyPlot := pd:-draw();
plotsetup(png, plotoutput=plotname, plotoptions="width=1024,height=768");
Export(plotname, MyPlot);
plotsetup(default);
printf("Plot saved to %s\n", plotname);'''



def get_ontology_column_names() -> list:
    fieldnames = []
    temp_string = ""
    for i in range(1, 1537):        #1536-d
        temp_string = f"embedding dim {i}"
        fieldnames.append(temp_string)

    fieldnames.append("function_value")
    return fieldnames

def get_df_from_csv(csv_path: str) -> pd.DataFrame:
    columns = get_ontology_column_names()
    df = pd.read_csv(csv_path, names=columns, header=0)
    print(df)
    return df

def run_maple(dataframe: pd.DataFrame, output_directory: str) -> list:
    os.mkdir("temp_maple_directory")
    dataframe.to_csv(path_or_buf="temp_maple_directory/temp_csv.csv", index = False)

    with open("temp_maple_directory/temp_maple_script.mpl", mode='w') as maple_script:
        maple_script.write(maple_script_str)
    
    
    with open("temp_maple_directory/log.txt", mode='w') as logger:
        logger.write("Beginning of log")
    log_path = Path("temp_maple_directory/log.txt")

    maple_process = subprocess.Popen(["maple", "temp_maple_directory/temp_maple_script.mpl"], stdout=log_path.open(mode="w"))

    plot_made = False
    while not plot_made:

        time.sleep(1)
        if "Plot saved to" in log_path.read_text():
            print("Temporary plot was saved")
            plot_made = True

    
    if plot_made:
        os.mkdir(output_directory)
        final_plot_path = output_directory + "/alpha_complex_plot.png"
        shutil.copyfile('temp_maple_directory/temp_alpha_complex_plot.png', final_plot_path)

        if "betti_numbers := [" in log_path.read_text():
            print("betti numa")
            with open("temp_maple_directory/log.txt", "r") as logger:
                logger_lines = logger.readlines()
                betti_found = False
                for i in range(len(logger_lines)):
                    if logger_lines[i] == '> printf("BETTI_NUMBER_MARKER");\n' and not betti_found:
                        betti_number_list = ""
                        start_adding = False
                        for char in logger_lines[i+4]:
                            if char == '[':
                                start_adding = True
                            if start_adding:
                                betti_number_list = betti_number_list + char
                        betti_number_list = ast.literal_eval(betti_number_list)

                        betti_found = True

    else:
        print("Plot was not saved")


    os.remove("temp_maple_directory/temp_alpha_complex_plot.png")
    os.remove("temp_maple_directory/log.txt")
    os.remove("temp_maple_directory/temp_csv.csv")
    os.remove("temp_maple_directory/temp_maple_script.mpl")
    os.rmdir("temp_maple_directory")
    print("All Finished")

    return betti_number_list


# def UmaskNamedTemporaryFile(*args, **kargs):
#     fdesc = tempfile.NamedTemporaryFile(*args, **kargs)
#     # we need to set umask to get its current value. As noted
#     # by Florian Brucker (comment), this is a potential security
#     # issue, as it affects all the threads. Considering that it is
#     # less a problem to create a file with permissions 000 than 666,
#     # we use 666 as the umask temporary value.
#     umask = os.umask(0o666)
#     os.umask(umask)
#     os.chmod(fdesc.name, 0o666 & ~umask)
#     return fdesc



# def run_maple(dataframe: pd.DataFrame, output_directory: str) -> None:
#     with tempfile.TemporaryDirectory() as temp_directory:
#         temp_csv = tempfile.NamedTemporaryFile(dir=temp_directory, suffix=".csv")
#         dataframe.to_csv(path_or_buf=temp_csv.name, index = False)
#         temp_mpl = UmaskNamedTemporaryFile(dir=temp_directory, suffix=".mpl")
#         temp_mpl.write(maple_script)
#         temp_log = tempfile.NamedTemporaryFile(dir=temp_directory, suffix=".txt")
#         log_path = Path(temp_log.name)

#         testing = subprocess.Popen(["maple", temp_mpl.name], stdout=log_path.open(mode="w"))
        




# testing = subprocess.Popen(["maple", "pdws1_modified_copy_ontology.mpl"])

# testing = subprocess.Popen(["maple", "pdws1_modified_copy_ontology.mpl", "-I", "data/aeo_terms.csv"])

# time.sleep(5)

# testing.terminate()
# os.rmdir("testing")

if __name__ == "__main__":
    my_df = get_df_from_csv("data/apo_terms.csv")
    print(my_df)

    nums = run_maple(my_df, "finaltesting")
    