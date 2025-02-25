# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st

# def display_violin_plots(df):
#     """
#     Displays violin plots for each column in the dataframe.
#     """
#     df = df.select_dtypes(include=['number'])

#     numeric_columns = df.columns
#     num_plots = len(numeric_columns)
    
#     # Divide into rows of 4 columns
#     rows = (num_plots + 3) // 4  # Compute the number of rows needed
#     for i in range(rows):
#         cols = st.columns(4)  # Create 4 columns
#         for j in range(4):
#             idx = i * 4 + j
#             if idx < num_plots:  # Check if index is within bounds
#                 column = numeric_columns[idx]
#                 with cols[j]:  # Add plot to the specific column
#                     plt.figure(figsize=(6, 4))
#                     sns.violinplot(data=df, y=column)
#                     plt.title(f"Violin Plot for {column}")
#                     st.pyplot(plt)
#                     plt.close()
    


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def convert_to_numeric(df):
    """
    تلاش برای تبدیل تمام مقادیر دیتافریم به عددی.
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # تبدیل به عدد، مقادیر نامعتبر NaN می‌شوند
    return df

def display_violin_plots(df):
    """
    نمایش ویولین پلات برای هر ستون عددی در دیتافریم.
    """
    df = convert_to_numeric(df)  # تبدیل تمام داده‌ها به عددی
    df = df.dropna(axis=1, how='all')  # حذف ستون‌هایی که کاملاً NaN هستند

    numeric_columns = df.columns
    num_plots = len(numeric_columns)
    
    if num_plots == 0:
        st.warning("هیچ داده عددی معتبری برای نمایش وجود ندارد!")
        return
    
    rows = (num_plots + 3) // 4  # تعداد ردیف‌های لازم برای نمایش ۴ نمودار در هر ردیف

    for i in range(rows):
        cols = st.columns(4)  # ایجاد ۴ ستون در هر ردیف
        for j in range(4):
            idx = i * 4 + j
            if idx < num_plots:  # بررسی اینکه ایندکس معتبر است
                column = numeric_columns[idx]
                with cols[j]:  # قرار دادن هر نمودار در ستون مخصوص خود
                    plt.figure(figsize=(6, 4))
                    sns.violinplot(data=df, y=column)
                    plt.title(f"Violin Plot for {column}")
                    st.pyplot(plt)
                    plt.close()
