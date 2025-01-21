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

AoA_p = -1
J_p = -1
delta_e_p = -1
V_inf_p = -1
Acoustic_p = -1
Time_p = -1
n_block = 1
block_names = iter([
    "Wind-off Measurements",
    "Main Block A",
    "Re Comparison",
    "Main Block B"
])

# Open the LaTeX file for writing
with open(latex_file, 'w') as output_file:
    # output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block}}} \\\\ \\hline \n")
    # n_block +=1
    # Loop through each row in the DataFrame and create the LaTeX rows
    for index, row in df.iterrows():
        AoA = row['AoA']
        J = row['J']
        delta_e = row['delta_a']
        V_inf = row['V_inf']
        Acoustic = row['Acoustic']
        Time = row['Time']
        
        Time_hms = convert_seconds_to_hms(Time)

        comment = ""

        if index != 0:
            if Acoustic == 1 and not (AoA == AoA_p and J == J_p and delta_e == delta_e_p):
                comment += "including acoustic measurement"
            elif Acoustic == 2:
                comment += "validation point"
            else:
                comment += ""

            if delta_e != delta_e_p:
                comment += "change in elevator and tunnel setting"
                block_name = next(block_names)
                # output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block} - {block_name}}} \\\\ \\hline \n")
                output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block}}} \\\\ \\hline \n")
                n_block +=1
            elif V_inf != V_inf_p:
                comment += "change in tunnel setting"
                block_name = next(block_names)
                # output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block} - {block_name}}} \\\\ \\hline \n")
                output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block}}} \\\\ \\hline \n")
                n_block +=1
        else:
            block_name = next(block_names)
            # output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block} - {block_name}}} \\\\ \\hline \n")
            output_file.write(f"\\multicolumn{{7}}{{|c|}}{{Block {n_block}}} \\\\ \\hline \n")
            n_block +=1

        if AoA == AoA_p and J == J_p and delta_e == delta_e_p:
            comment = "repetition point"


        rps = V_inf / J / D_prop if J != 0 else 0
        Jstring = f"{J:.2f}" if J!=0 else "N/A"
        
        # Write the row to the LaTeX file
        output_file.write(f"{index+1} & {V_inf:.0f} & {rps:.2f} ({Jstring}) & {AoA:.1f} &{delta_e:.1f} & {Time_hms} & {comment} \\\\ \\hline \n")

        AoA_p = AoA
        J_p = J
        delta_e_p = delta_e
        V_inf_p = V_inf
        Acoustic_p = Acoustic
        Time_p = Time