import numpy as np
import pandas as pd

data = np.array([[57,53],[37,18],[81,59],[35,45],[33,77]])
column = ["temperature", "activity"]
dataframe = pd.DataFrame(data=data, columns=column)
print(dataframe)

dataframe["adjusted"] = dataframe["activity"] + 2
print(dataframe)
print("\nRows #0, #1, #2:")
print(dataframe.head(3), '\n')
print("\nRow #2:")
print(dataframe.iloc[[2]], '\n')
print("Rows #1, #2, and #3:")
print(dataframe[1:4], '\n')

data = np.random.randint(low=0, high=101, size=(3,4))
column = ["Eleanor", "Chidi", "Tahani", "Jason"]
dataframe = pd.DataFrame(data=data, columns=column)
print(data)
print(dataframe)
print(dataframe["Eleanor"][1])
dataframe["Janet"] = dataframe["Tahani"] + dataframe["Jason"]
print(dataframe)

print("Experiment with a reference:")
reference_to_df = dataframe
print("  Starting value of df: %d" % dataframe['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % reference_to_df['Jason'][1])
dataframe.at[1, 'Jason'] = dataframe['Jason'][1] + 5
print("  Updated df: %d" % dataframe['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])
print("Experiment with a true copy:")
copy_of_my_dataframe = dataframe.copy()
print("  Starting value of my_dataframe: %d" % dataframe["Tahani"][1])
print("  Starting value of copy_of_my_dataframe: %d\n" % copy_of_my_dataframe["Tahani"][1])
dataframe.at[1, "Tahani"] = dataframe["Tahani"][1] + 3
print("  Updated my_dataframe: %d" % dataframe["Tahani"][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_my_dataframe["Tahani"][1])