from urllib.request import urlretrieve
import pandas as pd
def load_data(download=True):
    if download:
        data_path,_=urlretrieve("http://archive.ics.uci.edu/ml/"
                                "machine-learning-databases/car/car.data","car.csv")
        print("Downloaded to car.csv")
    col_names=["buying","maint","doors","persons","lug_bat","safety","class"]
    data =pd.read_csv("car.csv",names=col_names)
    return data
def convert2onehot(data):
    return pd.get_dummies(data,prefix=data.columns)
if __name__=='__main__':
    data =load_data(download=True)
    new_data =convert2onehot(data)
    print(data.head())
    print(" Num of data is:",len(data))
    #for name in data.keys():
       # print(name,pd.unique(data[name]))
    print(new_data.head(1))
    new_data.to_csv("car_hotone",index=False)