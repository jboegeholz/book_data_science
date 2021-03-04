## Pandas


    pip install pandas

    
    import pandas as pd 

### DataFrame


    df = pd.DataFrame()



```
df = pd.DataFrame({'age': [28, 35, 23],
                   'weight': [72, 88, 62],
                   'city':['Manhattan', 'Baltimore', 'Louisville']
                  },
                  index=['Peter', 'Paul', 'Mary'])
```              

### Excel
    df = pd.read_excel("path_to_excel_file")

    xl = pd.ExcelFile("path_to_excel_file")
    xl.sheet_names
    df = xl.parse("sheet_name")
    
    
    df.empty
True if NDFrame is entirely empty
You can get the column names via 
      
    df.columns
    
You can get an overview of the dataframe values with

    df.describe()
    
More information about the data frame 

    df.shape 
    df.dtypes 
    
You can access data from the dataframe via loc and iloc

    df.loc["Peter", 'age']

    df.iloc[0, 0]

the i in iloc stands for "index"

the colon is used as a wildcard for rows or columns
    
    df.loc[:, :]

#### Adding columns

    df.loc[:, 'new_column_name'] = new_column_data