"""
Boston Housing Prices Machine Learning Dashboard

Usage:
* pip3 install -r requirements.txt
* streamlit run main.py
"""
import plotly.express as px
import streamlit as st

import logging

from sklearn.datasets import load_boston

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title='James Davies Boston Housing Prices ML Dashboard',
    layout='wide'
)

st.sidebar.title('James Davies Boston Housing Prices ML Dashboard')
st.sidebar.write('This is a Machine Learning dashboard')

dashboard_function = st.sidebar.selectbox(
    label="Dashboard function",
    options=["Visualise", "Model"]
)


def load_data():
    """
    Load data

    Args:
        None

    Returns:
         df (DataFrame): Input boston dataset
    """
    LOGGER.info("Loading input data")
    # Load data and convert to pandas dataframe
    boston_dataset = load_boston()
    df = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])

    # Add target col
    df['target'] = boston_dataset['target']
    return df, boston_dataset['DESCR']


def _preprocessing(df):
    """
    Check for NULLs in input data

    Args:
        df (DataFrame)

    Returns:
        None
    """
    LOGGER.info("Preprocessing")
    null_df = df.isnull().sum()
    null_df = null_df[null_df > 0]

    if null_df.count() > 0:
        LOGGER.warning("There are NULLs present in the input data:")
        print(null_df)


def _data_exploration(df):
    """
    Calculate useful statistics

    Args:
        df (DataFrame)

    Returns:
        None
    """
    LOGGER.info("Calculating statistics")
    st.title("Please select col for statistical info:")
    target_col = st.selectbox('Col:', sorted(df.columns))
    target = df[target_col]
    st.title(f'Statistics for {target_col} column:')
    st.write(f'Min {target_col}: {np.amin(target)}')
    st.write(f'Max {target_col}: {np.amax(target)}')
    st.write(f'Mean {target_col}: {np.mean(target)}')
    st.write(f'Median {target_col}: {np.median(target)}')
    st.write(f'Standard deviation {target_col}: {np.std(target)}')


def _pick_cols_to_visualise(df):
    """
    Allow the user to choose what cols they'd like to visualise

    Args:
        df (DataFrame)

    Returns:
        cols_to_plot (List[String]) - list of user selected cols to plot
    """
    st.title("Choose cols to visualise")
    cols_to_plot = []
    for col in df.columns:
        plot = st.checkbox(f'Plot {col}?')
        if plot:
            cols_to_plot.append(col)
    return cols_to_plot


def _create_scatter_matrix(df, cols_to_plot):
    """
    Produce a scatter matrix
    Allow the user to select which cols to plot

    Args:
      df (Pandas DataFrame)
      cols_to_plot(List[String]) - list of cols to plot

    Returns:
      None
    """
    st.title('Scatter Matrix')
    st.plotly_chart(px.scatter_matrix(df[cols_to_plot]))


def _correlation_matrix(df, cols):
    """
    Produce a correlation matrix
    Allow the user to select which cols to plot

    Args:
        df (DataFrame)
        cols (List[String]) - cols to plot

    :param df:
    :return:
    """
    LOGGER.info("Creating correlation matrix")
    st.title('Correlation between features')
    if len(cols) < 2:
        st.write("Not enough cols selected for correlation matrix!")
    else:
        corr_matrix = np.corrcoef(df[cols].T)
        fig = plt.figure()
        heatmap = sns.heatmap(
            corr_matrix,
            fmt='.2f',
            annot=True,
            annot_kws={'size': 15},
            square=True,
            yticklabels=cols,
            xticklabels=cols
        )
        st.pyplot(fig)


def _plot_distribution(df):
    """
    Plot a distribution plot of the target column

    Args:
        df (DataFrame)

    Returns:
        None
    """
    st.title('Correlation between features')
    st.title("Please select col for correlation:")
    col = st.selectbox('Target Col:', sorted(df.columns))
    fig = plt.figure()
    sns.distplot(df[col], axlabel=col)
    st.pyplot(fig)


def visualise(df, data_desc):
    """
    Visualise entrypoint function

    Args:
        df (DataFrame) - input boston dataframe
        data_desc (string) - description of input data

    Returns:
        None
    """
    st.write(data_desc)
    _preprocessing(df)
    _plot_distribution(df)
    _data_exploration(df)
    cols_to_visualise = _pick_cols_to_visualise(df)
    _create_scatter_matrix(df, cols_to_visualise)
    _correlation_matrix(df, cols_to_visualise)


def _split_data(df, target_col):
    """
    Split data into testing and training

    Args:
      df (dataframe)
      target_col (string) - target column

    Returns:
        X_train
        X_test
        y_train
        y_test
    """
    test_size = st.sidebar.slider(
        label='Select the proportion of the input data to use for testing',
        min_value=0.05,
        max_value=0.95,
        step=0.05,
        value=0.3
    )
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target_col, axis=1).values,
                                                        df[target_col].values.ravel(),
                                                        test_size=test_size,
                                                        random_state=1)
    return X_train, X_test, y_train, y_test


def model(df):
    """
    Model entrypoint function

    Args:
        df (DataFrame)

    Returns:
        None
    """
    target_col = st.sidebar.selectbox('Select target column:', sorted(df.columns))
    X_train, X_test, y_train, y_test = _split_data(df, target_col)

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    y_train_predict = lin_model.predict(X_train)

    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2 = r2_score(y_train, y_train_predict)

    st.title("The model performance for training set")
    st.write(f'RMSE is {rmse}')
    st.write(f'R2 score is {r2}')


if __name__ == "__main__":
    LOGGER.info("Starting...")
    input_df, data_desc = load_data()
    if dashboard_function == 'Visualise':
        visualise(input_df, data_desc)
    elif dashboard_function == 'Model':
        model(input_df)
    LOGGER.info("Finished")
