# the following script is meant to be used for preprocessing a dataset of 50k samples per SDG based on Aurora queries
# apart from first (semi)manual steps - converting .csv files to .xlsx & removing soperflous columns from these files - all the other steps could be repeated running the script
# you just have to specify working directories (and name(s) of the files)


# all the targets
targets=["1.1", "1.2", "1.3", "1.4", "1.5", "1.a", "1.b",
        "2.1", "2.2", "2.3", "2.4", "2.5", "2.a", "2.b", "2.c",
        "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.a", "3.b", "3.c", "3.d",
        "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.a", "4.b", "4.c",
        "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.a", "5.b", "5.c",
        "6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.a", "6.b",
        "7.1", "7.2", "7.3", "7.a", "7.b",
        "8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9", "8.10", "8.a", "8.b",
        "9.1", "9.2", "9.3", "9.4", "9.5", "9.a", "9.b", "9.c",
        "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.a", "10.b", "10.c",
        "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.a", "11.b", "11.c",
        "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.a", "12.b", "12.c",
        "13.0", "13.1", "13.2", "13.3", "13.a", "13.b",
        "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.a", "14.b", "14.c",
        "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8", "15.9", "15.a", "15.b", "15.c",
        "16.1", "16.2", "16.3", "16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "16.10", "16.a", "16.b",
        "17.1", "17.2", "17.3", "17.4", "17.5", "17.6", "17.7", "17.8", "17.9", "17.10", "17.11", "17.12", "17.13", "17.14", "17.15", "17.16", "17.17", "17.18", "17.19"]


# remove empty superfluous columns (for each SDG file)
import pandas as pd

aurora_data=pd.read_excel("/.../....xlsx")
aurora_data=aurora_data[['EID', 'SDGs', 'Title', 'Abstract']]
aurora_data.to_excel("/.../....xlsx", index=False)


# filter nan EIDS, convert EIDS to integers (for each SDG file)
aurora_data=pd.read_excel("/.../....xlsx")
aurora_data=aurora_data[aurora_data["EID"].notna()]
aurora_data=aurora_data.astype({"EID": "int64"})
aurora_data.to_excel("/.../....xlsx", index=False)


# merge .xlsx aurora_datales and save the big one
import glob

files=glob.glob("/.../*.xlsx")
aurora_data=pd.DataFrame()
for _ in files:
    temp=pd.read_excel(_)
    aurora_data=aurora_data.append(temp)

aurora_data.to_excel("/.../SDGs_merged.xlsx", index=False)


# check for suspicious SDG labels and remove them
aurora_data=pd.read_excel("/.../SDGs_merged.xlsx")

from collections import Counter

counts=Counter(list(aurora_data.SDGs))
dcounts=dict(counts)
lcounts=list(dcounts.keys())
to_remove=[_ for _ in lcounts if len(_)>2]

aurora_data=aurora_data.astype({"SDGs": "str"})

for _ in to_remove:
    aurora_data=aurora_data[~aurora_data.SDGs.str.contains(_)]

aurora_data.to_excel("/.../SDGs_merged_cleaned.xlsx", index=False)


# extract targets and their correspondings eids per file in folder from folders - return dict object in shape: dict[target]: {eids}
import re

def read_eids(file, mode="r"):
    f=open(file).read()
    f=f.split("\n")
    f=[_[7:] for _ in f]
    f=f[:-1]
    f=[int(_) for _ in f]
    return f

def extract_targets_eids():
    output=dict()
    folders=glob.glob("/.../sdgs_eids/*/")
    for folder in folders:
        folder_files=glob.glob(folder+"/*txt")
        files=[_ for _ in folder_files if bool(re.match(r".*\d{1,2}\.{1}(\d{1,2}|\w{1}\_{1}).*", _)) == True]
        for file in files:
            eids=read_eids(file)
            target=re.findall(r"(\d{1,2}\.{1}\d{1,2}|\d{1,2}\.\w{1})",file)
            target="".join(target)
            output[target]=eids
    return output

targets_eids=extract_targets_eids()


# make one hot ecnodings per targets
for row_idx, row in aurora_data.iterrows():
    matches=[key for key in targets_eids.keys() if row.EID in targets_eids[key]]
    if len(matches) > 0:
        col_idxs=[aurora_data.columns.get_loc(_) for _ in matches]
        for col_idx in col_idxs:
            aurora_data.iloc[row_idx, col_idx]=1


# remove zero label rows
def remove_zero_label_rows(dataframe):
    temp=[]
    for _ in range(len(dataframe)):
        if sum(dataframe.iloc[_, 4:])>0:
            temp.append(dataframe.iloc[_,:])
    output=pd.DataFrame(temp, columns=dataframe.columns).reset_index(drop=True)
    return output

aurora_data=remove_zero_label_rows(aurora_data)


# remove rows where Abstract == NAN
aurora_data=aurora_data[aurora_data["Abstract"].isnull()==False]


# remove rows where Abstract is not a string
temp=[]
for _ in range(len(aurora_data)):
    if isinstance(aurora_data.iloc[_, 3], str):
        temp.append(aurora_data.iloc[_,:])

aurora_data=pd.DataFrame(temp, columns=aurora_data.columns).reset_index(drop=True)


# drop duplicates on Abstract, Title, EID
aurora_data=aurora_data.drop_duplicates(subset=["Abstract"])
aurora_data=aurora_data.drop_duplicates(subset=["Title"])
aurora_data=aurora_data.drop_duplicates(subset=["EID"])


# save the final file
aurora_data.to_hdf("/.../SDGs_merged_cleaned_onehot_no_zeros_no_duplicates.h5", key="SDGs_merged_cleaned_onehot_no_zeros_no_duplicates", mode="w")

