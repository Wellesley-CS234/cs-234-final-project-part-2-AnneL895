
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

st.title("CS234 Final Project")
st.subheader("Analyzing the Most Viewed Articles in March 2023 for 5 Countries ")
st.write("For this project, I decided to look at the most viewed articles for the top 5 countries that use Wikipedia. Those countries include the US, Japan, the UK, India, and Germany.")

st.write("First, let's take a look at all the pageviews over time for each country")

st.write("This is what my dataframe looks like:")
#load my dataframe
df = pd.read_csv("final-project-data2.csv")

#Line graph with the pageviews for the top articles over the course of the month
#Interactive part: be able to select countries to look at
st.write(df.head())

countries = ["US", "JP", "IN", "DE", "GB"]

selected_country = st.multiselect(
    "Now, let's pick a country to focus on",
    options = countries,
)

countrydf = df[df["country_code"].isin(selected_country)]
countrydf = countrydf.sort_values(by='date', ascending=False)
for_later = df.groupby(['date', "country_code", "qid"])['pageviews'].sum().reset_index()
countrydf = countrydf.groupby(['date', "country_code"])['pageviews'].sum().reset_index()

#what is the type of the data column in my dataframe?
countrydf["date"] = pd.to_datetime(countrydf["date"])

fig = px.line(
    countrydf,
    x="date",
    y="pageviews",
    color="country_code",   # separate line for each country
    markers=True,
    title="Daily Pageviews per Country (March 2023)"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Pageviews",
    hovermode="x unified"
)

st.title("Total Pageviews per Country for March")
st.plotly_chart(fig, use_container_width=True)


pageviews_df = for_later.groupby(["country_code"])['pageviews'].sum().reset_index()
#bar graph time
fig2 = px.bar(
    pageviews_df, 
    x='country_code', 
    y='pageviews', 
    title='Pageview count per country for the top ___ articles for the month of March 2023',
)

st.title("Pageviews per Country per Day")
st.plotly_chart(fig2, use_container_width=True)

#AVERAGE THIS BY POPULATION


#I am going to move on to other stuff rn because I am having issue with the classifier

humans_df = for_later[(for_later['country_code'] == 'US')]

json_df = pd.read_json("entity_results2.jsonl", lines=True)
json_df["instance_of"] = json_df["attributes"].apply(
    lambda x: x.get("instance of") if isinstance(x, dict) else np.nan
)


st.write("I am still working on generating visualizations for my text classification, but right now I have the pageviews for the US for articles about humans")
json_df = json_df.rename(columns={'QID': 'qid'})


to_merge = json_df[["qid", "instance_of"]]
merged = humans_df.merge(to_merge, on="qid", how="left")

#st.write(merged.head())
#but I just want the ones about humans and I just want to average them by day

humans = merged[merged['instance_of'] == 'human']
humans = humans.groupby(["date", "country_code"])['pageviews'].sum().reset_index()

countries_humans = humans[humans["country_code"].isin(selected_country)]


fig3 = px.line(
    countries_humans, 
    x='date', 
    y='pageviews', 
    color="country_code",
    markers=True,
    title='Pageview count per country for articles about humans for the month of March 2023',
)

st.title("Pageviews per Country for Top Articles about Humans per Day")
st.plotly_chart(fig3, use_container_width=True)

#I haven't labelled any country besides US, so rn this only works for US
