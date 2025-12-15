
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("CS234 Final Project")
st.subheader("Analyzing the Most Viewed Articles in March 2023 for 5 Countries ")
st.write("For this project, I decided to look at the most viewed articles for the top 5 countries that use Wikipedia. Those countries include the US, Japan, the UK, India, and Germany.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Basic Pageview Data", "Text Classification", "Hypothesis Testing", "Summary & Ethical Considerations"])

with tab1:
    st.title("Introduction")
    st.subheader("My Research Question:")
    st.write("For this project ...")
    st.subheader("My Research Question:")
    st.subheader("My Hypothesis:")
    st.write("This is where my hypothesis will go")
    st.subheader("Data Summary:")
    st.markdown("For this project, I wanted to focus on the top 5 countries that interact with Wikipedia articles:")
    st.markdown("- The United States")
    st.write("- Japan")
    st.write("- Great Britain")
    st.write("- India")
    st.write("- Germany")
    st.write("And to make this data more managable, I decided to only look at the top 5000 articles for each country for the month of March 2023.")
    st.write("For articles from the US, GB, and India, most articles were written in English, so I was able to use the facebook/bart-large-mnli text classifier on those data sets.")
    st.write("For the German articles, I used ...")
    st.write("And for the Japanese articles, I used ...")
    
    #Provide some descriptive statistics for the features of your dataset.
    st.subheader("New Features:")
    st.write("For this project, I used API calls to Wikidata in order to get labels for all the qids I collected from my articles")
    st.write("In total, I got wikidata on around 18,000 qids which took around 10 hours")
    st.write("My first text classification included labelling articles as person, place, event, or tv - the most common labels I saw in my wikidata under 'instance of' - in order to evaluate the accuracy on the models I was using")
    st.write("I was especially intersted in how the text classifier models would do when faced with a character based langauge like Japanese")
    st.write("Based on my findings from this preliminary text classification, I next wanted to see _____, and I was able to use the wikidata I collected and the text classifiers I was using to get the information I needed")
    

with tab2:
    st.write("First, let's take a look at all the pageviews over time for each country")

    st.write("This is what my dataframe looks like:")
    #load my dataframe
    df = pd.read_csv("basic_pageview_data_fig1,2.csv")

    st.write(df.head())

    countries = ["US", "JP", "IN", "DE", "GB"]

    selected_country = st.multiselect(
        "Now, let's pick a country to focus on",
        options = countries,
    )

    countrydf = df[df["country_code"].isin(selected_country)]
    countrydf = countrydf.sort_values(by='date', ascending=False)
    #for_later = df.groupby(['date', "country_code", "qid"])['pageviews'].sum().reset_index()
    #countrydf = countrydf.groupby(['date', "country_code"])['pageviews'].sum().reset_index()
    st.write(countrydf.head())

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


    pageviews_df = df.groupby(["country_code"])['pageviews'].sum().reset_index()
    #bar graph time
    fig2 = px.bar(
        pageviews_df, 
        x='country_code', 
        y='pageviews', 
        title='Pageview count per country for the top ___ articles for the month of March 2023',
    )

    st.title("Pageviews per Country for the Month of March")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("While it is interesting to look at the total pageviews per country, comparing data for different countries this way doesn't give us the most accurate picture of Wikipedia engagement accross countries. Let's divide by the population of each country to standardize our data.")
    standardize = st.toggle("standardize")

    if standardize:
        countrydf.loc[countrydf['country_code'] == 'US', 'pageviews'] = countrydf['pageviews'] / 342970425
        countrydf.loc[countrydf['country_code'] == 'JP', 'pageviews'] = countrydf['pageviews'] / 122809409
        countrydf.loc[countrydf['country_code'] == 'GB', 'pageviews'] = countrydf['pageviews'] / 69739901
        countrydf.loc[countrydf['country_code'] == 'IN', 'pageviews'] = countrydf['pageviews'] / 1469779612
        countrydf.loc[countrydf['country_code'] == 'DE', 'pageviews'] = countrydf['pageviews'] / 83859120
        fig_standardize = px.line(
            countrydf,
            x="date",
            y="pageviews",
            color="country_code",   # separate line for each country
            markers=True,
            title="Daily Pageviews per Country (March 2023) by Percent of Population"
        )

        fig_standardize.update_layout(
            xaxis_title="Date",
            yaxis_title="Pageviews",
            hovermode="x unified"
        )

        st.title("Total Pageviews per Country Based on Population for March ")
        st.plotly_chart(fig_standardize, use_container_width=True)


        pageviews_df = df.groupby(["country_code"])['pageviews'].sum().reset_index()
        pageviews_df.loc[pageviews_df['country_code'] == 'US', 'pageviews'] = pageviews_df['pageviews'] / 342970425
        pageviews_df.loc[pageviews_df['country_code'] == 'JP', 'pageviews'] = pageviews_df['pageviews'] / 122809409
        pageviews_df.loc[pageviews_df['country_code'] == 'GB', 'pageviews'] = pageviews_df['pageviews'] / 69739901
        pageviews_df.loc[pageviews_df['country_code'] == 'IN', 'pageviews'] = pageviews_df['pageviews'] / 1469779612
        pageviews_df.loc[pageviews_df['country_code'] == 'DE', 'pageviews'] = pageviews_df['pageviews'] / 83859120
        
        #bar graph time
        fig_bar_standardize = px.bar(
            pageviews_df, 
            x='country_code', 
            y='pageviews', 
            title='Pageview Count Per Country Based on Population for the Top 5000 Articles for the Month of March 2023',
            )

        st.title("Pageviews per Country Based on Population for the Month of March")
        st.plotly_chart(fig_bar_standardize, use_container_width=True)

        st.write("These graphs give us better insight into the interaction with Wikipedia by different countries as a whole")

with tab3:
    countries2 = ["US", "JP", "IN", "DE", "GB"]

    selected_country2 = st.multiselect(
        "Now, let's pick a country to focus on",
        options = countries2,
        key = "multiselect_tab2"
    )
    
    humans_grouped = pd.read_csv("text_classification_figures.csv")


    st.write("I am still working on generating visualizations for my text classification, but right now I have the pageviews for the US for articles about humans")
    #humans_grouped = humans_df.groupby(["date", "country_code"])['pageviews'].sum().reset_index()
    st.write(humans_grouped.head())

    countries_humans = humans_grouped[humans_grouped["country_code"].isin(selected_country2)]


    fig_humans = px.line(
        countries_humans, 
        x='date', 
        y='pageviews', 
        color="country_code",
        markers=True,
            title='Pageview count per country for articles about humans for the month of March 2023',
    )

    st.title("Pageviews per Country for Top Articles about Humans per Day")
    st.plotly_chart(fig_humans, use_container_width=True)

    st.header("This is where I am working now")
    #First I want to graph the percentages of person, place, event, or tv for each country
    #then I want to focus in on the people articles for each country and see if they are about pop culture or historical figures
    st.write("OMG, another cool visualization:")

    # Example: Load your CSVs for different countries
    US_df = pd.read_csv('US_daily_category_pct.csv', parse_dates=['date']).set_index('date')
    JP_df = pd.read_csv('JP_daily_category_pct.csv', parse_dates=['date']).set_index('date')
    GB_df = pd.read_csv('GB_daily_category_pct.csv', parse_dates=['date']).set_index('date')
    IN_df = pd.read_csv('IN_daily_category_pct.csv', parse_dates=['date']).set_index('date')
    DE_df = pd.read_csv('DE_daily_category_pct.csv', parse_dates=['date']).set_index('date')

    # Put them in a dictionary for easy access
    country_dfs = {
        "US": US_df,
        "JP": JP_df,
        "GB": GB_df,
        "IN": IN_df,
        "DE": DE_df
    }

    selected_country4 = st.selectbox(
        "Now, let's pick a country to focus on",
        countries2,
        key = "select_box2"
    )
    df_pct = country_dfs[selected_country4]
    
    
    #graphing part
    st.subheader("Custom Matplotlib Plot")
    fig_pct, ax = plt.subplots(figsize=(12, 6))
    df_pct.plot.area(ax=ax, cmap='tab20', alpha=0.8)
    ax.set_ylabel("Percentage of Pageviews")
    ax.set_xlabel("Date")
    ax.set_title("Daily Pageviews by Article Category (Percentage)")
    ax.legend(title="Category")
    st.pyplot(fig_pct)

    #Next figure
    st.header("Now it's time to focus on my last part of this project")
    st.subheader("I want to use my text classifier to label whether these people are related to pop culture and currect events or whether they are historical")

    humans_classification = pd.read_csv("human_classification_daily_category_pct.csv")
    st.write(humans_classification.head())

    countries2 = humans_classification['country_code'].unique() 
    
    selected_country5 = st.selectbox(
        "Now, let's pick a country to focus on",
        countries2,
        key="select_box3"
    )
    classification_to_graph = humans_classification[humans_classification["country_code"] == selected_country5]

    classification_to_graph = classification_to_graph.set_index('date')
    st.write(classification_to_graph.head())

    category_columns = ['historical', 'pop culture']

    st.subheader("Classification Plot")
    fig_classification, ax = plt.subplots(figsize=(12, 6))
    classification_to_graph[category_columns].plot.area(ax=ax, cmap='tab20', alpha=0.8)
    ax.set_ylabel("Percentage of Articles")
    ax.set_xlabel("Date")
    ax.set_title("Daily Pageviews by Article Category (Percentage)")
    ax.legend(title="Category")
    st.pyplot(fig_classification)

    st.write(humans_classification.head())
    avg_historical = humans_classification['historical'].mean()
    avg_pop_culture = humans_classification['pop culture'].mean()

    # If you only have one month or want the latest month:

    st.subheader(f"Metrics for {selected_country5}")

    st.metric("Percent Pop Culture", f"{avg_pop_culture:.3f}")
    st.metric("Percent Historical ", f"{avg_historical:.3f}")


with tab4:
    st.header("Do I need this page?")

with tab5:
    st.header("Summary and Ethical Considerations")
    st.subheader("What are the takeaways from your investigation? ")
    st.write("What are some limitations of your approach? How confident are you that the results are reliable? Are there any ethical concerns about this research? Did you expose any biases in the human activity data?")

    st.subheader("Zero-Shot Learning Classification Accuracy:")
    st.write("I evaluated my zero-shot learning classifier on the initial classification of data since I had the ground truth for those entries from the wikidata and could easily assess the accuracy")
    st.write("Here are the results I got from that:")
    st.write("* I have the data I need to evaluate my zero shot learning, I just need to write up that code and then put it here with my confusion matrices and such")
#I haven't labelled any country besides US, so rn this only works for US

    st.subheader("Visualizations for Assessing Zero Shot Learning")
    country_stats = pd.read_csv("country_stats.csv")
    st.write(country_stats)

    selected_country3 = st.selectbox(
        "Now, let's pick a country to focus on",
        country_stats["country"].unique()
    )

    normalize = st.toggle("Normalize confusion matrix", value=False)

    row = country_stats[country_stats["country"] == selected_country3].iloc[0]

    cm = np.array([
        [row["TP"], row["FN"]],
        [row["FP"], row["TN"]]
    ], dtype=float)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"
        cm = cm.astype(int)

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Person", "Not Person"],
        yticklabels=["Person", "Not Person"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    title = f"Confusion Matrix — {selected_country3}"
    if normalize:
        title += " (Normalized)"

    ax.set_title(title)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Metrics")

    st.metric("Precision", f"{row['precision']:.3f}")
    st.metric("Recall", f"{row['recall']:.3f}")
    st.metric("Accuracy", f"{row['accuracy']:.3f}")


