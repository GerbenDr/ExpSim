import pandas as pd
from datetime import timedelta

# Path to the input CSV file and the output LaTeX file
csv_file = 'matrix_w_time.csv'  # Replace with the path to your CSV file
latex_file = 'table_matrix.tex'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

D_prop = 0.2032  # m

# Function to convert seconds to hh:mm:ss format
def convert_seconds_to_hms(seconds):
    return str(timedelta(seconds=seconds))

# Open the LaTeX file for writing
with open(latex_file, 'w') as output_file:
    
    # Loop through each row in the DataFrame and create the LaTeX rows
    for index, row in df.iterrows():
        AoA = row['AoA']
        J = row['J']
        delta_e = row['delta_a']
        V_inf = row['V_inf']
        Acoustic = row['Acoustic']
        Time = row['Time']
        
        Time_hms = convert_seconds_to_hms(Time)
        # Determine the comment based on the Acoustic value
        comment = "acoustic" if Acoustic == 1 else "N/A"

        rps = V_inf / J / D_prop if J != 0 else 0
        Jstring = str(J) if J!=0 else "N/A"
        
        # Write the row to the LaTeX file
        output_file.write(f"{index+1} & {V_inf:.0f} & {rps:.2f} ({Jstring}) & {AoA:.1f} &{delta_e:.1f} & {Time_hms} & {comment} \\\\ \\hline \n")