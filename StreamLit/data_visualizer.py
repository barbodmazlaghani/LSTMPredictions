import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def display_violin_plots(df):
    """
    Displays violin plots for each column in the dataframe.
    """
    df = df.select_dtypes(include=['number'])

    numeric_columns = df.columns
    num_plots = len(numeric_columns)
    
    # Divide into rows of 4 columns
    rows = (num_plots + 3) // 4  # Compute the number of rows needed
    for i in range(rows):
        cols = st.columns(4)  # Create 4 columns
        for j in range(4):
            idx = i * 4 + j
            if idx < num_plots:  # Check if index is within bounds
                column = numeric_columns[idx]
                with cols[j]:  # Add plot to the specific column
                    plt.figure(figsize=(6, 4))
                    sns.violinplot(data=df, y=column)
                    plt.title(f"Violin Plot for {column}")
                    st.pyplot(plt)
                    plt.close()
