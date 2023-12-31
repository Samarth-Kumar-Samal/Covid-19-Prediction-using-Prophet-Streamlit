import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from prophet.plot import plot_plotly
from prophet import Prophet
import streamlit as st

# Setting up the page configutation
st.set_page_config(
    page_title='Covid-19 Prediction Application',
    page_icon='💻',
    layout='wide'
)

# Setting up the page title
st.title('Covid-19 Prediction Web Application 📊')
st.write('Developing a COVID-19 prediction web app with Prophet for forecasting, Plotly for interactive visualizations, and Streamlit for a user-friendly interface.')

st.markdown("<br>", unsafe_allow_html=True)

# Data Collection
df = pd.read_csv('.\\dataset\\covid-19.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Set default values to the minimum and maximum dates in the dataset
default_start_date = pd.to_datetime(df['Date']).min()
default_end_date = pd.to_datetime(df['Date']).max()

# Create two columns for start date and end date
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(f"Input Start Date (Default : {default_start_date.date()})", value=default_start_date)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected Start Date:", start_date)

with col2:
    end_date = st.date_input(f"Input End Date (Default : {default_end_date.date()})", value=default_end_date)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Selected End Date:", end_date)

# Convert start_date and end_date to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Checking the correctness of the start date and end date
if start_date < default_start_date:
    st.warning('Invalid Start Date')
elif end_date > default_end_date:
    st.warning('Invalid End Date')

st.markdown("<br>", unsafe_allow_html=True)

# Displaying the filtered dataset
st.subheader('Filtered Dataset:')
dff = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
dff['Date'] = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
st.write(dff)

# Converting the Date column from object type to datetime type
dff['Date'] = pd.to_datetime(dff['Date'])

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('World Covid Cases with respect to time')

st.markdown("<br>", unsafe_allow_html=True)

# Create an expander for the choropleth map of Confirmed Cases
with st.expander("Confirmed Cases with respect to Time"):
    # Create choropleth map
    fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color='Confirmed',
        animation_frame='Date',
        color_continuous_scale='RdBu',
    )
    # Update layout
    fig.update_layout(
        title='Choropleth Map for the total number of Confirmed Covid-19 Cases around the world'
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
    # Show the map using Streamlit
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

# Create an expander for the choropleth map of Recovered Cases
with st.expander("Recovered Cases with respect to Time"):
    # Create choropleth map
    fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color='Recovered',
        animation_frame='Date',
        color_continuous_scale='BuPu',
    )
    # Update layout
    fig.update_layout(
        title='Choropleth Map for the total number of Recovered Covid-19 Cases around the world'
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
    # Show the map using Streamlit
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

# Create an expander for the choropleth map of Recovered Cases
with st.expander("Death Cases with respect to Time"):
    # Create choropleth map
    fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color='Deaths',
        animation_frame='Date',
        color_continuous_scale='magma',
    )
    # Update layout
    fig.update_layout(
        title='Choropleth Map for the total number of Death Covid-19 Cases around the world'
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
    # Show the map using Streamlit
    st.plotly_chart(fig)


st.markdown("<br>", unsafe_allow_html=True)

# Death Cases

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Top 5 countries for Confirmed,Deaths and Recovered Covid-19 cases')

with st.expander(label='Top 5 countries having maximum number of Covid-19 Death Cases') :
    deaths_by_country = dff.groupby('Country')['Deaths'].sum()
    desc = deaths_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(5)]
    filtered_data = dff[dff['Country'].isin(country_list)]

    fig = px.bar(
        filtered_data,
        x="Country",
        y="Deaths",
        color="Country",
        animation_frame="Date",
        title="Death Cases of top 5 countries",
        range_y=[0, filtered_data['Deaths'].max() + 100000]
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.5
    st.plotly_chart(fig)


st.markdown("<br>", unsafe_allow_html=True)

with st.expander(label='Top 5 countries having maximum number of Covid-19 Confirmed Cases') :
    confirmed_by_country = dff.groupby('Country')['Confirmed'].sum()
    desc = confirmed_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(5)]
    filtered_data = dff[dff['Country'].isin(country_list)]

    fig = px.bar(
        filtered_data,
        x="Country",
        y="Confirmed",
        color="Country",
        animation_frame="Date",
        title="Confirmed Cases of top 5 countries",
        range_y=[0, filtered_data['Confirmed'].max() + 100000]
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.5
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander(label='Top 5 countries having maximum number of Covid-19 Recovered Cases') :
    recovered_by_country = dff.groupby('Country')['Recovered'].sum()
    desc = recovered_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(5)]
    filtered_data = dff[dff['Country'].isin(country_list)]

    fig = px.bar(
        filtered_data,
        x="Country",
        y="Recovered",
        color="Country",
        animation_frame="Date",
        title="Recovered Cases of top 5 countries",
        range_y=[0, filtered_data['Confirmed'].max() + 100000]
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.5
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Top 10 most affected countries with confirmed cases, recovered cases and death cases')

col1,col2,col3 = st.columns(3)

with col1 :
    deaths_by_country = df.groupby('Country')['Deaths'].sum()
    desc = deaths_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(10)]
    filtered_data = df[df['Country'].isin(country_list)]
    fig=px.pie(filtered_data,values='Deaths',names='Country',title="Percentage of Total Death Cases in 10 most affect countries")
    st.plotly_chart(fig)

with col2:
    confirmed_by_country = df.groupby('Country')['Confirmed'].sum()
    desc = confirmed_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(10)]
    filtered_data = df[df['Country'].isin(country_list)]
    fig=px.pie(filtered_data,values='Confirmed',names='Country',title="Percentage of Total Confirmed Cases in 10 most affect countries")
    st.plotly_chart(fig)

with col3:
    recovered_by_country = df.groupby('Country')['Recovered'].sum()
    desc = recovered_by_country.sort_values(ascending=False)
    country_list = [desc.index[i] for i in range(10)]
    filtered_data = df[df['Country'].isin(country_list)]
    fig=px.pie(filtered_data,values='Recovered',names='Country',title="Percentage of Total Recovered Cases in 10 most affect countries")
    st.plotly_chart(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Forecasting the Covid-19 Cases')

st.markdown("<br>", unsafe_allow_html=True)

country = st.selectbox(label='Select the Country Name',options=dff['Country'].unique())
year = st.slider(label='Select the number of years', min_value=1, max_value=10)
period = year * 365

with st.expander(label=f'Forecasting the Confirmed Cases for {country}'):
    confirmed_data = dff[dff['Country'] == country][['Date', 'Confirmed']]
    confirmed_data['Date'] = pd.to_datetime(confirmed_data['Date'])
    confirmed_data.rename(columns={'Date': 'ds', 'Confirmed': 'y'}, inplace=True)
    fig = px.line(confirmed_data, x='ds', y='y', title="Time Series Graph for Confirmed Cases")
    st.write(fig)
    model = Prophet()
    model.fit(confirmed_data)
    future_pred = model.make_future_dataframe(periods=period, freq='D')  # Specify frequency as 'D' for daily
    prediction = model.predict(future_pred)
    st.write(prediction)
    fig2 = model.plot_components(prediction)
    st.write(fig2)
    fig3 = plot_plotly(model,prediction,xlabel='Time',ylabel='Confirmed cases')
    fig3.update_layout(title='Forecast graph of Time Series model')
    st.write(fig3)

with st.expander(label=f'Forecasting the Death Cases for {country}'):
    deaths_data = dff[dff['Country'] == country][['Date', 'Deaths']]
    deaths_data['Date'] = pd.to_datetime(deaths_data['Date'])
    deaths_data.rename(columns={'Date': 'ds', 'Deaths': 'y'}, inplace=True)
    fig = px.line(deaths_data, x='ds', y='y', title="Time Series Graph for Deaths Cases")
    st.write(fig)
    model = Prophet()
    model.fit(deaths_data)
    future_pred = model.make_future_dataframe(periods=period, freq='D')  # Specify frequency as 'D' for daily
    prediction = model.predict(future_pred)
    st.write(prediction)
    fig2 = model.plot_components(prediction)
    st.write(fig2)
    fig3 = plot_plotly(model,prediction,xlabel='Time',ylabel='Death cases')
    fig3.update_layout(title='Forecast graph of Time Series model')
    st.write(fig3)

with st.expander(label=f'Forecasting the Recovered Cases for {country}'):
    recovered_data = dff[dff['Country'] == country][['Date', 'Recovered']]
    recovered_data['Date'] = pd.to_datetime(recovered_data['Date'])
    recovered_data.rename(columns={'Date': 'ds', 'Recovered': 'y'}, inplace=True)
    fig = px.line(recovered_data, x='ds', y='y', title="Time Series Graph for Recovered Cases")
    st.write(fig)
    model = Prophet()
    model.fit(recovered_data)
    future_pred = model.make_future_dataframe(periods=period, freq='D')  # Specify frequency as 'D' for daily
    prediction = model.predict(future_pred)
    st.write(prediction)
    fig2 = model.plot_components(prediction)
    st.write(fig2)
    fig3 = plot_plotly(model,prediction,xlabel='Time',ylabel='Recovered cases')
    fig3.update_layout(title='Forecast graph of Time Series model')
    st.write(fig3)