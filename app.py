import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# preparing for the header
st.set_page_config(page_title='Google play store',
                   layout='wide', initial_sidebar_state="auto")
img2 = "https://zenkimchi.com/wp-content/uploads/2015/11/google-play-googleplay-button-logo-download-on-black-glossy-jane-bordeaux-2-copy.png"
img3 = "https://play-lh.googleusercontent.com/SfpWypzOwUw3R-yQgX6Z2O79wZ5KeMS9GCIAGIfAErEf86ybMCsw93gpPpuKvTaMDWg"
# Define the CSS style to center the image and set its width and height
style1 = (
    "display: block;"
    "margin: 0 auto;"
    f"width: 180px;"
    f"height: 65px;"
)
style2 = (
    "display: block;"
    "margin: 0 auto;"
    f"width: 65px;"
    f"height: 65px;"
)
# Display the image with the specified style
col00, col01, col02 = st.columns(3)
with col00:
    st.markdown(f'<img src="{img3}" style="{style2}">', unsafe_allow_html=True)
with col01:
    st.header("Google play apps")
with col02:
    st.markdown(f'<img src="{img2}" style="{style1}">', unsafe_allow_html=True)
# displaying the header
# headter complete


# reading the file
df = pd.read_csv('google/googleplaystored.csv')
pd.DataFrame(df)
# Styling page (using css)
st.markdown(
    """
    <style>
       h1,
h2,
.css-k7vsyb span,
.css-10y5sf6{
  color: rgb(186, 131, 239);
}
div.st-cc > div > div > div > div:nth-child(2) > input[type="range"]::-webkit-slider-thumb {
background-color: rgb(124, 29, 213); 
}
div.st-cc > div > div > div > div:nth-child(2) > input[type="range"]::-webkit-slider-runnable-track {
background-color: rgb(96, 21, 165);
}
.st-br        ,.st-cn{
background: rgb(124, 29, 213); 
}
.css-1vzeuhh{
background-color:rgb(62, 15, 106);
}
.css-1dp5vir{
background-image: linear-gradient(90deg, rgb(44 2 78), rgb(169 149 255));
height: 10px;
}
.css-tvhsbf .glidedf1Editor{
 box-shadow: 115px 115px 115px 0 rgba(0,0,0,0.2);
border:2px solid rgb(62, 15, 106);
}
    </style>
    
    """,

    unsafe_allow_html=True
)


# to clean data . Actually there was an empty string spaces , so it causing errors
df['Reviews'] = df['Reviews'].apply(lambda x: int(
    float(x.strip('M').replace('Varies with device', '0').replace(',', ''))))


# Navigation bar
st.sidebar.header("Select Ranges here:")
# To fire qurries according to the specific data range
min_rows, max_rows = st.sidebar.slider(
    "Select Number of Rows Range",
    min_value=1,
    max_value=len(df),
    value=(1, len(df))
)
min_reviews, max_reviews = st.sidebar.slider(
    "Select Reviews Range",
    min_value=df["Reviews"].min(),
    max_value=df["Reviews"].max(),
    value=(df["Reviews"].min(), df["Reviews"].max())
)
min_rating, max_rating = st.sidebar.slider(
    "Select rating Range",
    min_value=df["Rating"].min(),
    max_value=df["Rating"].max(),
    value=(df["Rating"].min(), df["Rating"].max())
)


# Cleaning installs data
def clean_and_convert_installs(value):
    try:
        # Remove non-numeric characters and convert to integer
        return int(''.join(filter(str.isdigit, value)))
    except (ValueError, TypeError):
        return 0


# to prepare atributed columns for qurrey fire
df['Installs'] = df['Installs'].apply(clean_and_convert_installs)
min_inst, max_inst = st.sidebar.slider(
    "Select Installs Range",
    min_value=df["Installs"].min(),
    max_value=df["Installs"].max(),
    value=(df["Installs"].min(), df["Installs"].max())
)
Category = st.sidebar.multiselect(
    "Select the Category:",
    options=df["Category"].unique(),
    default=df["Category"].unique(),
)
Type = st.sidebar.multiselect(
    "Select the Type:",
    options=df["Type"].unique(),
    default=df["Type"].unique(),
)
Content_rating = st.sidebar.multiselect(
    "Select the content rating :",
    options=df["Content Rating"].unique(),
    default=df["Content Rating"].unique(),
)
Genres = st.sidebar.multiselect(
    "Select the Genres :",
    options=df["Genres"].unique(),
    default=df["Genres"].unique(),
)
# Print selected values for debugging
print("\n\nSelected category values:\n", Category)
print("\n\nSelected Type values:\n", Type)
print("\n\nSelected Content_Rating values:\n", Content_rating)
print("\n\nSelected Grenes values:\n", Genres)
print("\n\nSelected Reviews Range:\n", min_reviews, max_reviews)
print("\n\nSelected ratings Range:\n", min_rating, max_rating)
print("\n\nSelected install Range:\n", min_inst, max_inst)
print("\n\nSelected Number of Rows Range:\n", min_rows, max_rows)


# showing data on app
st.title("Google Play Store Dashboard")
st.write("Filtered df1:")
try:
    df1 = df.query("Category == @Category & Type == @Type & `Content Rating` == @Content_rating & Genres == @Genres & @min_reviews <= Reviews <= @max_reviews & @min_rating <= Rating <= @max_rating & @min_rows <= Index <= @max_rows")
    df1 = df1.iloc[min_rows - 1:max_rows]
    st.dataframe(df1)
except ValueError as ve:
    st.error(f"1ValueError: {ve}")


# showing the facts . Numerical analysis
st.markdown("#")
numeric_columns = df1[['Rating', 'Reviews', 'Installs', 'Price']]
total_reviews = df1["Reviews"].sum()
average_review = round(df1["Reviews"].mean(), 2)
noofapps = len(df1)
avgrat = df1['Rating'].mean()
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Reviews : ")
    st.markdown(f"Total Reviews: {total_reviews:,}", unsafe_allow_html=True)
    st.markdown("<style>h3 { font-size: 16px; }</style>",
                unsafe_allow_html=True)
with middle_column:
    st.subheader("Number of apps")
    st.markdown(f"Number of apps {noofapps:,}", unsafe_allow_html=True)
    st.markdown("<style>h3 { font-size: 16px; }</style>",
                unsafe_allow_html=True)
with right_column:
    st.subheader("Average rating :")
    st.markdown(f"Average rating: {avgrat:,}", unsafe_allow_html=True)
    st.markdown("<style>h3 { font-size: 16px; }</style>",
                unsafe_allow_html=True)


mostdown = df1["Installs"].max()

col11, col12 = st.columns(2)
with col11:
    most_downloaded_app = ""
    most_downloads = ""
    try:
        # Calculate the most downloaded app
        most_downloaded_app = df1.loc[df1['Installs'].idxmax()]['App']
        most_downloads = df1['Installs'].max()
        least_downloaded_app = df1.loc[df1['Installs'].idxmin()]['App']
        least_downloads = df1['Installs'].min()
        # Calculate the most reviewed app
        most_reviewed_app = df1.loc[df1['Reviews'].idxmax()]['App']
        most_reviews = df1['Reviews'].max()
        least_reviewed_app = df1.loc[df1['Reviews'].idxmin()]['App']
        least_reviews = df1['Reviews'].min()
        # Calculate the most rated app
        most_rated_app = df1.loc[df1['Rating'].idxmax()]['App']
        highest_rating = df1['Rating'].max()
        least_rated_app = df1.loc[df1['Rating'].idxmin()]['App']
        least_rating = df1['Rating'].min()
    except ValueError as ve:
        st.write("No df1 available for the ranged apps")
    st.title("Facts")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.subheader("Most Downloaded App:")
            st.markdown(
                f"App Name: {most_downloaded_app}", unsafe_allow_html=True)
            st.markdown(
                f"Total Downloads: {most_downloads:,}", unsafe_allow_html=True)
            st.subheader("least Downloaded App:")
            st.markdown(
                f"App Name: {least_downloaded_app}", unsafe_allow_html=True)
            st.markdown(
                f"Total Downloads: {least_downloads:,}", unsafe_allow_html=True)

        with col2:
            st.subheader("Most Reviewed App:")
            st.markdown(f"App Name: {most_reviewed_app}",
                        unsafe_allow_html=True)
            st.markdown(
                f"Total Reviews: {most_reviews:,}", unsafe_allow_html=True)
            st.subheader("least Reviewed App:")
            st.markdown(f"App Name: {least_reviewed_app}",
                        unsafe_allow_html=True)
            st.markdown(
                f"Total Reviews: {least_reviews:,}", unsafe_allow_html=True)
        with col3:
            st.subheader("Highest Rated App:")
            st.markdown(f"App Name: {most_rated_app}", unsafe_allow_html=True)
            st.markdown(
                f"Highest Rating: {highest_rating}", unsafe_allow_html=True)
            st.subheader("least Rated App:")
            st.markdown(f"App Name: {least_rated_app}", unsafe_allow_html=True)
            st.markdown(
                f"least Rating: {least_rating}", unsafe_allow_html=True)
    except ValueError as ve:
        st.write("No df1 available for the ranged apps")
with col12:
    category_grouped = df.groupby('Category')
    category_stats = category_grouped.agg(
        Number_of_Ratings=pd.NamedAgg(column='Reviews', aggfunc='count'),
        Average_Rating=pd.NamedAgg(column='Rating', aggfunc='mean'),
        Number_of_Reviews=pd.NamedAgg(column='Reviews', aggfunc='sum'),
        Average_Reviews=pd.NamedAgg(column='Reviews', aggfunc='mean'),)
    st.dataframe(category_stats)


# plotting top 10 bar graphs for subgraphs
top_10_apps = df1.nlargest(10, 'Installs')
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].bar(top_10_apps['Category'],
              top_10_apps['Installs'], color='skyblue')
axs[0, 0].set_title('Installs')
axs[0, 1].bar(top_10_apps['Category'],
              top_10_apps['Reviews'], color='lightgreen')
axs[0, 1].set_title('Reviews')
axs[1, 0].bar(top_10_apps['Category'], top_10_apps['Rating'], color='orange')
axs[1, 0].set_title('Rating')
axs[1, 1].bar(top_10_apps['Category'], top_10_apps['Category'],
              color='pink')  # Just for demonstration
axs[1, 1].set_title('Category')


# pie chart preparation
plt.tight_layout()
#  pie chart for the top 10 most reviewed apps
fig1, ax = plt.subplots()
ax.pie(top_10_apps['Reviews'], labels=top_10_apps['App'],
       autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
ax.set_title('Top 10 Most Reviewed Apps')
#  pie chart for the top 10 most installed apps
fig2, ax = plt.subplots()
ax.pie(top_10_apps['Installs'], labels=top_10_apps['App'],
       autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
ax.set_title('Top 10 Most Installed Apps')

st.header("Pie chart share for top 10 apps")
pie1, pie2 = st.columns(2)
with pie1:
    st.pyplot(fig1)

with pie2:
    st.pyplot(fig2)

# Machine learning algorithm
category_encoder = LabelEncoder()
content_rating_encoder = LabelEncoder()
category_encoder.fit(df['Category'])
df['Category'] = category_encoder.transform(df['Category'])
content_rating_encoder.fit(df['Content Rating'])
df['Content Rating'] = content_rating_encoder.transform(df['Content Rating'])
category_mapping = dict(zip(category_encoder.classes_,
                        category_encoder.transform(category_encoder.classes_)))
content_rating_mapping = dict(zip(content_rating_encoder.classes_,
                              content_rating_encoder.transform(content_rating_encoder.classes_)))
# Train a Linear Regression model
X = df[['Category', 'Content Rating', 'Reviews']]
y = df['Rating']
model = LinearRegression()
model.fit(X, y)
st.title("Google Play Store NEW App Rating Prediction")
st.subheader("Predict App Rating")
user_category = st.selectbox(
    "Select App Category:", list(category_mapping.keys()))
user_content_rating = st.selectbox(
    "Select Content Rating:", list(content_rating_mapping.keys()))
user_reviews = st.number_input("Enter Number of Reviews:", min_value=0)
# Map user input values
user_category_encoded = category_mapping[user_category]
user_content_rating_encoded = content_rating_mapping[user_content_rating]
# Create a DataFrame for the user input
user_data = pd.DataFrame(
    [[user_category_encoded, user_content_rating_encoded, user_reviews]])
# Make predictions for the user input
predicted_rating = model.predict(user_data)
st.write(f"Predicted Rating for the App: {predicted_rating[0]:.2f}")


# to view top rank holders for sorting
st.title("Top Rank Holders :")
col1, col2, col3 = st.columns(3)
sorted_by_installs = df1.sort_values(by='Installs', ascending=False)
sorted_by_ratings = df1.sort_values(by='Rating', ascending=False)
sorted_by_reviews = df1.sort_values(by='Reviews', ascending=False)
with col1:
    st.subheader("Top Apps by Installs:")
    st.dataframe(sorted_by_installs[['App', 'Installs']].head(10))
with col2:
    st.subheader("Top Apps by Reviews:")
    st.dataframe(sorted_by_reviews[['App', 'Reviews']].head(10))
with col3:
    st.subheader("apps by ratings:")
    st.dataframe(sorted_by_ratings[['App', 'Rating']].head(10))

st.header("Bar chart subplots for top 10 apps")
st.pyplot(fig)


df = pd.DataFrame(df1)
top_apps = df.sort_values('Reviews', ascending=False).groupby(
    'Category').head(1).head(10)
fig3, ax = plt.subplots(figsize=(12, 4))


# Plotting the stacked bar graph
bar_width = 0.35
bar_positions = list(range(len(top_apps)))
bars1 = ax.bar(bar_positions, top_apps['Installs'],
               bar_width, color='skyblue', label='Installs')
bars2 = ax.bar([p + bar_width for p in bar_positions],
               top_apps['Reviews'], bar_width, color='lightgreen', label='Reviews')
bars3 = ax.bar([p + 2 * bar_width for p in bar_positions],
               top_apps['Rating'], bar_width, color='orange', label='Rating')
ax.set_xlabel('Category')
ax.set_ylabel('Counts')
ax.set_title('Top App in Each Category: Installs, Reviews, and Ratings')
ax.set_xticks([p + bar_width for p in bar_positions])
ax.set_xticklabels(top_apps['Category'])
plt.grid()
ax.legend()
st.header("Bar chart for top apps")
st.pyplot(fig3)


# Create subplots for scatter plots
st.header("Scatter chart for top apps")
fig5, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].scatter(df['Installs'], df['Reviews'], color='skyblue')
axs[0].set_title('Installs vs. Reviews')
axs[0].set_xlabel('Installs')
axs[0].set_ylabel('Reviews')
axs[1].scatter(df['Installs'], df['Rating'], color='lightgreen')
axs[1].set_title('Installs vs. Ratings')
axs[1].set_xlabel('Installs')
axs[1].set_ylabel('Ratings')
axs[2].scatter(df['Reviews'], df['Rating'], color='orange')
axs[2].set_title('Reviews vs. Ratings')
axs[2].set_xlabel('Reviews')
axs[2].set_ylabel('Ratings')
plt.tight_layout()
st.pyplot(fig5)


# Create a 3D scatter plot
fig6 = plt.figure(figsize=(8, 6))
ax = fig6.add_subplot(111, projection='3d')
colors = np.random.rand(len(df))
ax.scatter(df['Installs'], df['Reviews'], df['Rating'], c=colors, marker='o')
ax.set_xlabel('Installs')
ax.set_ylabel('Reviews')
ax.set_zlabel('Ratings')
ax.set_title('3D Scatter Plot: Installs, Reviews, and Ratings')
st.pyplot(fig6)


# heatmap
st.header("Heatmap for apps")
df1['Reviews'] = pd.to_numeric(df1['Reviews'])
top_20_apps = df1.nlargest(20, 'Reviews')
heatmap_data = top_20_apps.pivot_table(
    index='App', columns='Rating', values='Reviews')
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g')
plt.title('Heatmap of Reviews and Ratings for Top 20 Apps')
st.pyplot(plt.gcf())

# line charts
st.header("Line charts for apps")
df1 = df1.sort_values('Reviews', ascending=False)
top_30_apps = df1.head(30)
plt.figure(figsize=(12, 6))
plt.plot(top_30_apps['App'], top_30_apps['Rating'], marker='o', label='Rating')
plt.plot(top_30_apps['App'], top_30_apps['Reviews'],
         marker='o', label='Reviews')
plt.xticks(rotation=45)
plt.xlabel('Apps')
plt.ylabel('Values')
plt.title('Top 30 Apps: Ratings and Reviews')
plt.legend()
st.pyplot(plt.gcf())


# stack plot
st.header("Stack plot for apps")
df1 = df1.sort_values('Reviews', ascending=False)
top_20_apps = df1.head(20)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.stackplot(top_20_apps['App'], top_20_apps['Reviews'],
              labels=['Reviews'], alpha=0.8, colors='r')
ax1.set_ylabel('Reviews')
ax1.set_title('Stack Plot of Reviews for Top 20 Apps')
ax2.stackplot(top_20_apps['App'], top_20_apps['Rating'],
              labels=['Ratings'], alpha=0.8, colors='g')
ax2.set_xlabel('Apps')
ax2.set_ylabel('Ratings')
ax2.set_title('Stack Plot of Ratings for Top 20 Apps')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())


st.header("Bar graph for Catagerywise apps")
top_apps = df.sort_values('Reviews', ascending=False).groupby(
    'Category').head(1).head(25)
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
axs[0].bar(top_apps['App'], top_apps['Installs'], color='skyblue')
axs[0].set_ylabel('Installs')
axs[0].tick_params(axis='x', rotation=45)
axs[1].bar(top_apps['App'], top_apps['Reviews'], color='lightgreen')
axs[1].set_ylabel('Reviews')
axs[1].tick_params(axis='x', rotation=45)
axs[2].bar(top_apps['App'], top_apps['Rating'], color='orange')
axs[2].set_ylabel('Ratings')
axs[2].tick_params(axis='x', rotation=45)
fig.suptitle('Top App in Each Category: Installs, Reviews, and Ratings',
             fontsize=16, fontweight='bold')
plt.tight_layout()
st.pyplot(plt.gcf())
