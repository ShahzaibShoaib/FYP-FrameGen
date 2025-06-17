import pandas as pd

def test(arr):
    series = pd.Series(arr)
    length = len(series)
    frequency = series.value_counts()
    unique_count = len(series.unique())
    min_value = min(frequency)  
    ans = length - (unique_count * min_value)  
    return ans

print(test([3, 3, 2, 1, 3]))           
print(test([3, 3, 3, 1, 1, 2, 2]))     
