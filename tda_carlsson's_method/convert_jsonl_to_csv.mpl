# This file will convert the jsonl files to csv for TDA

jsonl_to_csv := proc(jsonl_path, csv_path) 
    local jsonl_file, csv_file, line_in_file, json_object;

    # Open the JSONL file to read through it
    jsonl_file := fopen(jsonl_file, READ);

    # Open a CSV file to write to it
    csv_file := fopen(csv_path, WRITE);

    # Write the CSV header for new file
    fprintf(csv_file, ) # check columns for this 

    # Read JSONL file line by line 
    while not feof(jsonl_file) do
        lines := readline(jsonl_file);
        json_object := parseJSON(lines);
        fprintf(csv_file, "%a")

    end do;