import pandas as pd
class cleaner:
    def __init__(self, df, label, mean_columns="All", mode_columns=[]):
        self.df = df
        self.label = label
        self.zeros = self.df[self.df[self.label] == 0]
        self.ones = self.df[self.df[self.label] == 1]
        self.column_mean_zero = {}
        self.column_mean_one = {}
        self.column_mode_zero = {}
        self.column_mode_one = {}
        for column in self.df.columns:
            self.column_mean_zero[column] = self.zeros.loc[:,column].mean().round(3)
            self.column_mean_one[column] = self.ones.loc[:,column].mean().round(3)
            self.column_mode_zero[column] = self.zeros.loc[:,column].mode()[0].round(3)
            self.column_mode_one[column] = self.ones.loc[:,column].mode()[0].round(3)
        if mean_columns == "All":
            self.mean_columns = self.df.columns
            self.mode_columns = []
        else:
            self.mean_columns = mean_columns
            self.mode_columns = mode_columns
        for column in self.mean_columns:
            self.zeros[column].fillna(self.column_mean_zero[column], inplace = True)
            self.ones[column].fillna(self.column_mean_one[column], inplace = True)
        for column in self.mode_columns:
            self.zeros[column].fillna(self.column_mode_zero[column], inplace = True)
            self.ones[column].fillna(self.column_mode_one[column], inplace = True)
        frames = [self.zeros, self.ones]
        self.result = pd.concat(frames)
        self.result = self.result.sample(frac=1).reset_index(drop=True)
        
if __name__ == "__main__":
    df = pd.read_csv ('data.csv')
    label = "reslt"
    clean = cleaner(df, label, ['a', 'age', 'weight1', 'HB', 'IFA', 'BP1', 'res'], ['history'])
    print(clean.result)
    clean.result.round(3)
    clean.result.to_csv("cleaned_data.csv", index = None, header=True)