
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("CS234 Final Project")
st.subheader("Analyzing the Most Viewed Articles in March 2023 for 5 Countries ")
st.write("For this project, I decided to look at the most viewed articles for the top 5 countries that use Wikipedia. Those countries include the US, Japan, Great Britain (Wikipedia groups England, Scotland, and Wales), India, and Germany.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Basic Pageview Data", "Text Classification", "Hypothesis Testing", "Summary & Ethical Considerations"])

full_countries = {
    "US": "United States",
    "JP": "Japan",
    "GB": "Great Britain",
    "IN": "India",
    "DE": "Germany"
}

with tab1:
    st.title("Introduction")
    st.subheader("My Research Question:")
    st.write("For this project I initially started by looking at the top articles for the top 5 countries that use Wikipedia.")
    st.write("When I got the Wikidata for these articles, I noticed that the top categories for articles were person, place, event, or tv/film. From there, I decided to test 3 zero-shot classifiers by having them label Wikipedia articles just based on their names as one of the four top categories.")
    st.write("(You can see the results of this step on my Summary and Ethical Condiserations tab)")
    st.write("But I had the ground truth for all of these articles from the Wikidata, so I could have labelled them myself. To do some real text classification, I decided to focus in on the articles about people, since these made up the majority of articles accross countries.")
    st.write("My research question ended up being: Do Wikipedia users from the top 5 countries that interact with Wikipedia prefer to read articles about historical figures or people featured in current pop culture?")


    st.subheader("My Hypothesis:")
    st.write("My hypothesis was that the top Wikipedia using countries would be most interested in people featured in current pop culture.")
    
    st.subheader("Data Summary:")
    st.markdown("For this project, I wanted to focus on the top 5 countries that interact with Wikipedia articles, namely:")
    st.markdown("- The United States")
    st.write("- Japan")
    st.write("- Great Britain *(England, Scotland, and Wales)")
    st.write("- India")
    st.write("- Germany")
    st.write("And to make this data more managable, I decided to only look at the top 5000 articles for each country for the month of March 2023.")
    st.write("For articles from the US, GB, and India, most articles were written in English, so I was able to use the facebook/bart-large-mnli text classifier on those data sets.")
    st.write("For the German articles, I used the joeddav/xlm-roberta-large-xnli classifier")
    st.write("And for the Japanese articles, I used the MoritzLaurer/mDeBERTa-v3-base-mnli-xnli classifier")
    st.write("Data Statistice:")
    st.metric("Unique QIDs", "16101")
    st.metric("Countries", "5")

    
    #Provide some descriptive statistics for the features of your dataset.
    st.subheader("New Features:")
    st.write("For this project, I used API calls to Wikidata in order to get labels for all the qids I collected from my articles.")
    st.write("In total, I got wikidata on around 18,000 qids which took around 14 hours total.")
    st.write("My first text classification included labelling articles as about a person, place, event, or as related to tv/film  (the most common labels I saw in my wikidata under 'instance of') in order to evaluate the accuracy on the models I was using.")
    st.write("Based on my findings from this preliminary text classification, I labelled all my data one of the 4 categories.")
    st.write("From there, I focused on the articles about people and used the wikidata I had gathered about all my qids to find descriptions for each person. Next, I used the text classifier I used earlier for English language text classification to now label my articles about humans as either historical or current figures. Because the wikidata was in English, I converted my Japanese and German label into English to make the process easier.")

    st.write("I hope you enjoy exploring the data I collected over the course of this project and my findings!")
    

with tab2:
    st.subheader("Let's take a some pageview data")

    st.write("First, this is what my dataframe looks like:")
    #load my dataframe
    df = pd.read_csv("basic_pageview_data_fig1,2.csv")

    st.write(df.head())

    countries = ["US", "JP", "IN", "DE", "GB"]

    selected_country = st.multiselect(
        "Now, let's pick a country to focus on:",
        options = countries,
    )

    countrydf = df[df["country_code"].isin(selected_country)]
    countrydf = countrydf.sort_values(by='date', ascending=False)

    #what is the type of the data column in my dataframe?
    countrydf["date"] = pd.to_datetime(countrydf["date"])

    fig = px.line(
        countrydf,
        x="date",
        y="pageviews",
        color="country_code",   
        markers=True,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Pageviews",
        hovermode="x unified",
        title = "Daily Pageviews per Country for the Top 5000 Articles for March 2023"
    )

    st.plotly_chart(fig, use_container_width=True)


    pageviews_df = df.groupby(["country_code"])['pageviews'].sum().reset_index()
    #bar graph time
    fig2 = px.bar(
        pageviews_df, 
        x='country_code', 
        y='pageviews', 
        title = "Total Pageviews per Country for the Top 5000 Articles for March 2023"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.write("While it is interesting to look at the total pageviews per country, comparing data for different countries this way doesn't give us an accurate picture of Wikipedia engagement accross countries. Let's divide by the population of each country to standardize our data.")
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
            color="country_code",  
            markers=True,
        )

        fig_standardize.update_layout(
            xaxis_title="Date",
            yaxis_title="Pageviews",
            hovermode="x unified",
            title = "Daily Pageviews per Country Based on Population for Top 5000 Articles for March 2023"
        )

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
            title='Total Pageviews per Country Based on Population for the Top 5000 Articles for the Month of March 2023',
            )

        st.plotly_chart(fig_bar_standardize, use_container_width=True)

        st.write("These graphs give us better insight into the interaction with Wikipedia by different countries as a whole.")

        st.write("We can also take a look at what the top articles were:")

        topdf = pd.read_csv("top25articles_per_country.csv")

        st.write(topdf)
        

with tab3:
    st.write("Now that we have all our pageview data, we can focus in on the classifiers I chose: person, place, event, tv/film, as well as the subgroups of historical vs pop culture for humans.")

    countries2 = ["US", "JP", "IN", "DE", "GB"]

    selected_country2 = st.multiselect(
        "Pick a country to focus on",
        options = countries2,
        key = "multiselect_tab2"
    )
    
    humans_grouped = pd.read_csv("text_classification_figures.csv")

    countries_humans = humans_grouped[humans_grouped["country_code"].isin(selected_country2)]


    fig_humans = px.line(
        countries_humans, 
        x='date', 
        y='pageviews', 
        color="country_code",
        markers=True,
        title='Daily Pageviews per Country for Articles About Humans for the Month of March 2023',
    )

    st.plotly_chart(fig_humans, use_container_width=True)

    #First I want to graph the percentages of person, place, event, or tv for each country
    #then I want to focus in on the people articles for each country and see if they are about pop culture or historical figures
    st.write("While it is interesting to generally look at the pageviews per country for articles about humans, it's difficult to contextualize that information. Let's compare the number of articles about humans for each country to the number of articles about other classifiers.")

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
        "Pick a country to focus on",
        countries2,
        key = "select_box2"
    )
    df_pct = country_dfs[selected_country4]


    
    
    #graphing part
    st.write(f"Daily Pageviews by Article Category (Percentage) for {full_countries[selected_country4]}")
    fig_pct, ax = plt.subplots(figsize=(12, 6))
    df_pct.plot.area(ax=ax, cmap='tab20', alpha=0.8)
    ax.set_ylabel("Percentage of Pageviews")
    ax.set_xlabel("Date")
    ax.legend(title="Category")
    st.pyplot(fig_pct)

    #Next figure
    st.write("Now that we've explored some data related to our first 4 classifiers, let's take a closer look at the articles about humans.")
    
    humans_classification = pd.read_csv("human_classification_daily_category_pct.csv")

    countries2 = humans_classification['country_code'].unique() 
    
    selected_country5 = st.selectbox(
        "Pick a country to focus on",
        countries2,
        key="select_box3"
    )
    classification_to_graph = humans_classification[humans_classification["country_code"] == selected_country5]

    classification_to_graph = classification_to_graph.set_index('date')

    category_columns = ['historical', 'pop culture']

    st.subheader(f"Daily Pageviews by Article Category (Percentage) for {full_countries[selected_country5]}")
    fig_classification, ax = plt.subplots(figsize=(12, 6))
    classification_to_graph[category_columns].plot.area(ax=ax, cmap='tab20', alpha=0.8)
    ax.set_ylabel("Percentage of Articles")
    ax.set_xlabel("Date")
    ax.legend(title="Category")
    st.pyplot(fig_classification)

    avg_historical = humans_classification['historical'].mean()
    avg_pop_culture = humans_classification['pop culture'].mean()

    # If you only have one month or want the latest month:

    st.subheader(f"Metrics for {full_countries[selected_country5]}")

    st.metric("Pop Culture", f"{avg_pop_culture:.3f}")
    st.metric("Historical ", f"{avg_historical:.3f}")

    st.write("We can also take a look at the top articles about humans for each country:")

    tophumansdf = pd.read_csv("top25articles_humans.csv")

    st.write(tophumansdf)



with tab4:
    st.header("Anlyzing Results")
    st.write("From the text classification page, we compared the different between the percent of articles about historical figures and the percent of articles about pop culture figures. Below is the same visualization and statistics from the last page.")
    selected_country6 = st.selectbox(
        "Pick a country to focus on",
        countries2,
        key="select_box4"
    )
    classification_to_graph2 = humans_classification[humans_classification["country_code"] == selected_country6]

    st.subheader(f"Daily Pageviews by Article Category (Percentage) for {full_countries[selected_country6]}")
    fig_classification2, ax = plt.subplots(figsize=(12, 6))
    classification_to_graph2[category_columns].plot.area(ax=ax, cmap='tab20', alpha=0.8)
    ax.set_ylabel("Percentage of Articles")
    ax.set_xlabel("Date")
    ax.legend(title="Category")
    st.pyplot(fig_classification2)

    avg_historical2 = humans_classification['historical'].mean()
    avg_pop_culture2 = humans_classification['pop culture'].mean()

    st.subheader("Findings:")
    st.write("Although my hypothesis was that people would generally be more interested in reading about figures in pop culture in turns out that the majority of articles interacted with about humans were actually about historical figures.")
    st.write("This was quite suprising to me especially because of how strongly in favor of articles about historical figures the countries were.")
    st.write("This data doesn't support my hypothesis and suggests that people are more intersted in reading about histrorical figures then current figures (at least in these countries for the month of March 2023).")

with tab5:
    st.header("Summary and Ethical Considerations")
    st.subheader("What are the takeaways from your investigation? ")
    st.write("What are some limitations of your approach? How confident are you that the results are reliable? Are there any ethical concerns about this research? Did you expose any biases in the human activity data?")
    st.write("Limitations")
    st.write("I think that some limitations of this approach are first the fact that I only looked a one month from one year to get my data. By only looking at this small snapshot of data, it's difficult to know whether this is a general trend or if this month was an outlier for people interacting with articles about historical figures.")
    st.write("To get a more accurate gauge of popular articles topics, I think that pulling multiple weeks or days from different points in the year or over multiple years could provide a better assessment of the trends I saw.")

    st.write("Reliability")
    st.write("Because the accuracy of the text classification models I used fluctuated between models and countries, I am not sure how reliable my findings for the second part of my text classification are. I had the ground truth for my first round of text classification, so I was able to see how accurate those labels were, but I didn't have time to conduct an evaluation of the text classification for historical vs pop culture figures since I would need to label the articles myself based on the wikidata descriptions.")
    st.write("For this reason, I am not sure how accurate my results are or how much they can really tell us about peoples' interaction with Wikipedia articles.")

    st.write("Ethical Concerns and Bias:")
    st.write("One thing that was particularly interesting for me to see in this research was that for India, most of the top articles were in English. We discussed how this was most likely do to the fact that the people in India who are using Wikipedia are usually those of a higher education level and how English is widely spoken in India because it is used in government, business, and higher education.")
    st.write("This also relates to how, because Wikipedia is a US based organization, it makes sense that it would have more American users, leading to higher pageviews for the US as well as more articles (and specifically more articles in English).")
    st.write("Overall, I think the most bias in the data comes down to this surplus of data for the US while there is much less data for other countries.")

    st.subheader("Zero-Shot Learning Classification Accuracy:")
    st.write("I evaluated my zero-shot learning classifier on the initial classification of data since I had the ground truth for those entries from the wikidata and could easily assess the accuracy.")
    st.write("Here are the results I got from that:")
    
    st.subheader("Visualizations for Assessing Zero Shot Learning")
    country_stats = pd.read_csv("country_stats.csv")
    st.write("This is what my dataframe looks like:")
    st.write(country_stats)

    selected_country3 = st.selectbox(
        "Pick a country to focus on",
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

    title = f"Confusion Matrix — {full_countries[selected_country3]}"
    if normalize:
        title += " (Normalized)"

    ax.set_title(title)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Metrics")

    st.metric("Precision", f"{row['precision']:.3f}")
    st.metric("Recall", f"{row['recall']:.3f}")
    st.metric("Accuracy", f"{row['accuracy']:.3f}")

    st.write("As you can see, there are differences in the metrics for different countries and different classifiers, with the classification of articles from Great Britain having the worst accuracy. This was suprising to me since I used the same classifier on Great Britain, and US, and India, and the other two countries have a significantly higher accuracy. I am not sure why this is the case, and if I had more time, I think it would have been interesting to dive more into this aspect of the project to understand what happened with this text classifier.")


