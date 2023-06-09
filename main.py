import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import streamlit as st
import base64

# Background Image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

add_bg_from_local('Data/flight.jpg')

# Texts
st.title("Data Analysis for U.S. International Air Traffic data(1990-2020)")
st.sidebar.header("About")
st.sidebar.divider()
st.sidebar.subheader("Developed by: Priyanka Raparthi")
st.sidebar.markdown("Data and Computing Science Researcher - Carleton University")
st.sidebar.markdown("Contact: gouthamipriyanka1512@gmail.com")
st.sidebar.divider()
st.sidebar.markdown("Description")
st.sidebar.markdown("This app is 💡inspired from the use case-study https://resources.sw.siemens.com/hu-HU/case-study-natilus")
st.sidebar.divider()
st.sidebar.markdown("This App is to demonstrate the importance of data analysis and how data quality can impact the decisions we make. The explanations and complete documentation can be found in the github link at the end. There could be further analysis that can be conducted using this data such as predicting the airport usage in the upcoming years, relationship between carrier and airport, predicting the passengers count using the passengers data to determine flight prices and airfreight charges in the future. These can be performed using various Machine Learning techniques but require much more data preprocessing and transformations.")
st.sidebar.divider()
st.sidebar.markdown("This app is deployed using Streamlit Community Cloud")
st.sidebar.markdown("Link to dataset: https://data.transportation.gov/Aviation/International_Report_Departures/innc-gbgc")
st.sidebar.markdown("For complete code and explanation check https://github.com/gouthami-tom/Air_Traffic_Analysis")

# This code is to demonstrate the importance of data analysis and how data quality can impact the decisions we make.

# Load the data into a dataframe
df_original = pd.read_csv("Data/International_Report_Departures.csv")


# Print the summary
buffer = io.StringIO()
df_original.info(buf=buffer)
s = buffer.getvalue()

with st.container():
    st.header('Data Summary')
    st.text(s)

# We do not have the descriptions of the columns in the dataset.
# Create a dataframe using MultiIndex in Pandas with column descriptions from the source

columns = [('data_dte', 'Data Date'),
           ('Year', 'Data Year'),
           ('Month', 'Data Month'),
           ('usg_apt_id', 'US Gateway Airport ID - assigned by US DOT to identify an airport'),
           ('usg_apt',
            'US Gateway Airport Code - usually assigned by IATA, but in absence of IATA designation, may show '
            'FAA-assigned code'),
           ('usg_wac', 'US Gateway World Area Code - assigned by US DOT to represent a geographic territory'),
           ('fg_apt_id', 'Foreign Gateway Airport ID - assigned by US DOT to identify an airport'),
           ('fg_apt',
            'Foreign Gateway Airport Code - usually assigned by IATA, but in absense of IATA designation, '
            'may show FAA-assigned code'),
           ('fg_wac', 'Foreign Gateway World Area Code - assigned by US DOT to represent a geographic territory'),
           ('airlineid', 'Airline ID - assigned by US DOT to identify an air carrier'),
           ('carrier',
            'IATA-assigned air carrier code. If carrier has no IATA code, ICAO- or FAA-assigned code may be used'),
           ('carriergroup', 'Carrier Group Code - 1 denotes US domestic air carriers, 0 denotes foreign air carriers'),
           ('type', 'Defines the type of the four metrics in this report'),
           ('Scheduled', 'Metric flown by scheduled service operations'),
           ('Charter', 'Metric flown by charter operations'),
           ('Total', 'Total Metric flown by scheduled service and charter operations')]
#
df_descriptions = pd.DataFrame(df_original.values,
                               columns=pd.MultiIndex.from_tuples(columns, names=["Columns", "Column Description"]))


# Print the summary with descriptions added
def show_column_descriptions():
    with st.beta_expander("Column Descriptions"):
        st.write(df_descriptions)


if st.button("Show Column Descriptions"):
    show_column_descriptions()

# Note that the datatype of fields other than object from the original dataframe changed into object, resulting in a
# dataframe with all the values as object types. This is because the MultiIndex stores the data as pandas series
# which is of type object. We can perform typecasting here explicitly but in order not to complicate it let us use
# the original dataframe for further analysis as this step is to just know what the columns actually mean in the
# dataset.

# Print initial rows
st.subheader("First Look of the Data")
st.write('-----------------------------------------------------')
st.write(df_original.head())
st.write('-----------------------------------------------------')

# Print the shape(dimensions)
st.write('The dimensions of the data are', df_original.shape)

# The data is huge and hence let us consider a sample of the data for our analysis
# 0.1 indicates we consider 10% of the rows from original dataset
df_sampled = df_original.sample(frac=0.1, random_state=22)

st.write('The dimensions of the sampled data are', df_sampled.shape)
# It is very important that we do not consider a random sample, as the resultant analysis may be biased towards a
# particular feature. Hence, we need to know the descriptive statistics and distribution of the sampled dataframe for
# an accurate analysis.

# 1. Sample size
st.write('The size of the sampled dataframe is' , str(len(df_sampled)))

# 2. Sample Distribution
# For the distribution we can plot same variables from the original data and sampled data
# and check if their distribution is similar or not.
# Here let us consider the "Total" column for the distribution

# sns.set_style('darkgrid')
st.subheader("Data distribution comparision between the original and sample dataframes")
fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
sns.histplot(data=df_original, x="Total", kde=True, bins=5, ax=ax1)
sns.histplot(data=df_sampled, x="Total", kde=True, bins=5, ax=ax2)
st.pyplot(fig1)

# Both the distributions appear to be the same and hence we can proceed with our analysis with the sampled dataset

# Data Cleaning is an important step before performing any analysis on the data. Cleaning the data includes a series
# of steps.

# 1. Find out the missing values and perform data imputation
print('-----------------------------------------------------')
print('NULL VALUES')
print('-----------------------------------------------------')
print(df_sampled.isnull().sum())

# There are 300 null values in the carrier column. For better understanding, export the dataframe to an Excel and
# examine the null values. NaN means null values in terms of pandas. Although there can be different kinds of
# notations to represent missing values.

# Uncomment this line to generate the excel again
# df_sampled.to_excel("Data/sampled.xlsx", engine="openpyxl")

# When we filter the missing carrier values in the Excel, it is clear that of all the columns, the "airlineid" and
# "carriergroup" follow a specific pattern. Only the carriers with values "20414" and "20415" have missing carriers.
# Also, their "carriergroup" is 1 (US domestic air carriers).

# To fill the values, let us understand what the airlineid's mean.
# airlineid -> carrier
# 20414     -> "OW"
# 20415     -> "XG"
# This clearly tells we can replace the carriers with these respective ID's to eliminate null values in the data.
# But before replacing the null values, observe that they are blank spaces and not NaN.
# Hence, filter all the rows with missing values and fill the blank spaces with NaN first.
# Then replace with respective values.

df_missingcarriers = df_sampled.isnull().values.any(axis=1)

# Prints the rows that has any missing values in df_sampled
print(df_sampled[df_missingcarriers])

# Now fill the original dataframe with NaN for missing values.
df_sampled = df_sampled.replace(r'^\s*$', np.nan, regex=True)

# Fill the missing values with respective values
df_sampled.loc[df_sampled["airlineid"].isin([20414, 20415]), "carrier"] = df_sampled["airlineid"].map(
    {20414: "OW", 20415: "XG"})

# Calculate the missing values again.
print('-----------------------------------------------------')
print('NULL VALUES AFTER IMPUTATION')
print('-----------------------------------------------------')
print(df_sampled.isnull().sum())

# 2. Check for duplicate entries
print('-----------------------------------------------------')
print('DUPLICATE ENTRIES')
print('-----------------------------------------------------')
duplicates = df_sampled.duplicated()
if duplicates.any():
    print(duplicates.value_counts())
else:
    print("No duplicate entries found")

# 3. Examine the outliers
# To examine the outliers we can use different plots and see if we have extreme values
# that effect the overall distribution.

columns_of_interest = ["Scheduled", "Charter", "Total"]
df_outliers = df_sampled[columns_of_interest].copy()
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


st.subheader("Histogram and a Boxplot showing skewness and Outliers")
# histogram for distribution
ax1.hist(df_outliers, label=["Scheduled", "Charter", "Total"])
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Sampled Data Distribution')
ax1.legend()

# #boxplot
ax2.boxplot(df_outliers)
ax2.set_ylabel('Values')
ax2.set_xticklabels(["Scheduled", "Charter", "Total"])
ax2.set_title('Box Plot of Outlier Values')

st.pyplot(fig2)

# We can see that the data is positively skewed since the values are mostly on the left side of the distribution,
# indicating some extreme values in the distribution. This also means that we need to normalize the data or transform
# it before using any algorithms that assume a normally distributed data. The boxplot indicates the outlier values
# clearly and support the distribution in the histogram.


# Simple Insights using the dataset

# Busiest airports

_= """
This function is used to determine the busiest airports based on the area code 
by IATA and plot them using a barplot.

Parameters
----------
df: pandas dataframe
    Dataframe that needs to be plotted
groupby_column: plain_text
                Column name by which groupby should be performed
sum_column:    plain_text 
                Column name we want to visualize the groupby
title:  plain_text
        title for the plot

Returns
---------
Figure with the plots

"""

def busiest_airports(df, groupby_column, sum_column,title):
    departures_by_airport = df.groupby(groupby_column)[sum_column].sum()
    busiest_airports = departures_by_airport.sort_values(ascending=False)
    airport = busiest_airports.index[0:10]
    total = busiest_airports.values[0:10]
    fig, ax = plt.subplots()
    ax.bar(airport, total)
    ax.set_xlabel(groupby_column)
    ax.set_ylabel(sum_column)
    ax.set_title(title + sum_column)
    return fig


fig3 = busiest_airports(df_sampled, "usg_apt", "Total","Top 10 Busiest US Airports (1990-2020) based on ")
st.pyplot(fig3)
fig4 = busiest_airports(df_sampled,"fg_apt", "Total","Top 10 Busiest Foreign Airports (1990-2020) based on ")
st.pyplot(fig4)

# There could be further analysis that can be conducted using this data such as predicting the airport usage in the
# upcoming years, relationship between carrier and airport, predicting the passengers count using the passengers data
# to determine flight prices and airfreight charges in the future. These can be performed using various Machine
# Learning techniques but require much more data preprocessing and transformations.

# This code is to demonstrate the importance of data analysis and how data quality can impact the decisions we make.
