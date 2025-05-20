import streamlit as st
import streamlit.components.v1 as components
from PIL import Image  # For handling images
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob  # Ensure this is installed


# --- Basic Page Configuration ---
st.set_page_config(
    page_title="Mushthafa Aminur Rahman - Data Portfolio",
    page_icon=":bar_chart:",  # You can choose a different emoji
    layout="wide",
)
# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Homepage",
        "Data Visualization Projects",
        "Data Analysis Projects",
        "UI/UX Design Projects",  # Corrected page name
        "About Me",
        "Contact",
    ],
)

# --- Main Content Area ---
if page == "Homepage":
    # --- Hero Section ---
    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
    with col1:
        try:
            profile_image = Image.open(
                r"C:\Users\umaml\Downloads\c83083495df9fec9d3e55b61d071ee53.gif"
            )  # Replace with your image path
            st.image(profile_image, width=200)
        except FileNotFoundError:
            st.error("Profile image not found. Please update the path.")

    with col2:
        st.title("Mushthafa Aminur Rahman")
        st.subheader("Data Visualization Designer & Analyst | UI/UX Enthusiast")
        st.write(
            "Welcome to my online portfolio showcasing my work in data visualization, data analysis, and UI/UX design. Explore my projects below!"
        )
        st.markdown(
            "[LinkedIn](https://www.linkedin.com/in/mushthafa/) | [GitHub](your_github_url)"
        )  # Replace with your actual URLs

    st.markdown("---")

    # --- Featured Projects (Optional) ---
    st.subheader("Featured Projects")
    # You can display thumbnails or brief descriptions of a few key projects here
    # Example:
    col3, col4 = st.columns(2)
    with col3:
        try:
            project1_image = Image.open(
                r"C:\Users\umaml\Pictures\wallpapers\GNBzm-ObUAAekm1.jpg"
            )  # Replace with your image path
            st.image(project1_image, caption="Project 1 Title")
        except FileNotFoundError:
            st.error("Project 1 image not found. Please update the path.")
        st.markdown("[View Project 1](#project-1)")  # Create an anchor link later

    with col4:
        try:
            project2_image = Image.open(
                r"C:\Users\umaml\Downloads\1.png"
            )  # Replace with your image path
            st.image(project2_image, caption="Project 2 Title")
        except FileNotFoundError:
            st.error("Project 2 image not found. Please update the path.")
        st.markdown("[View Project 2](#project-2)")  # Create an anchor link later

    st.markdown("---")

    # --- Code Snippet Showcase ---
    st.subheader("Code Snippet Highlight")
    st.write(
        "Here's a glimpse into some of the Python code I use for data analysis and visualization:"
    )

    # Example Code Snippet 1
    st.code(
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        data = {'Category': ['A', 'B', 'C', 'D'],
                'Value': [25, 40, 30, 55]}
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        plt.bar(df['Category'], df['Value'])
        plt.xlabel('Category')
        plt.ylabel('Value')
        plt.title('Sample Bar Chart')
        plt.show()
        """,
        language="python",
    )

    st.write(
        "This snippet demonstrates creating a simple bar chart using Matplotlib with Pandas."
    )

    st.markdown("---")

    # Example Code Snippet 2
    st.code(
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Sample data (replace with your actual data)
        tips = sns.load_dataset("tips")

        sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)
        plt.title("Total Bill vs. Tip Amount")
        plt.show()
        """,
        language="python",
    )

    st.write(
        "This snippet showcases a scatter plot created with Seaborn, highlighting the relationship between total bill and tip amount."
    )

    st.markdown("---")

    # --- Stock Prediction Project Snippet on Homepage ---
    st.subheader("Featured Data Analysis Project: Stock Price Prediction")
    st.write("A project demonstrating stock price prediction using LSTM and Streamlit.")
    st.code(
        """
        import streamlit as st
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import plotly.graph_objects as go

        # --- Simplified Snippet ---
        st.subheader("Predicting BBRI Stock Price (Snippet)")
        ticker = "BBRI.JK"
        data = yf.download(ticker, period="1y")
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(close_prices)

        # Basic LSTM model (for demonstration)
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(None, 1)),
            LSTM(units=50),
            Dense(units=1)
        ])

        st.write("A simplified LSTM model is used for demonstration.")
        fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines')])
        fig.update_layout(title='Closing Price of BBRI (Last Year)', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
        """,
        language="python",
    )
    st.markdown("[View Full Project](#stock-prediction-project)")  # Link to the full project section

elif page == "Data Visualization Projects":
    st.header("Data Visualization Projects")
    st.write(
        "Showcase your data visualization projects here. For each project, you can include descriptions, images/interactive plots, code snippets, and links."
    )

elif page == "Data Analysis Projects":
    st.header("Data Analysis Projects")
    st.subheader("Stock Price Prediction Dashboard")
    st.write(
        "This project demonstrates the use of LSTM neural networks to predict the stock prices of BBRI, TLKM, and ANTM. Users can select a stock and the number of days to predict, and the dashboard displays the historical data, the predicted prices, and a sentiment analysis based on recent news and social media."
    )

    # Stock Prediction Dashboard Implementation
    stock_choice = st.selectbox("📌 Pilih Saham:", ["BBRI", "TLKM", "ANTM"])
    predict_days = st.selectbox("🧮 Jumlah hari yang ingin diprediksi:", [7, 14, 30])

    # Ticker mapping
    ticker_map = {
        "BBRI": "BBRI.JK",
        "TLKM": "TLKM.JK",
        "ANTM": "ANTM.JK",
    }
    tickers = ["BBRI.JK", "TLKM.JK", "ANTM.JK"]
    selected_ticker = ticker_map[stock_choice]
    selected_index = tickers.index(selected_ticker)

    # Download data
    @st.cache_data
    def get_data():
        df = pd.DataFrame()
        today = date.today()
        start_date = today - timedelta(days=5 * 365)  # 5 years ago. Account for leap years, roughly
        end_date = today
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
                if data is not None and not data.empty:  # Check if data is valid
                    df[ticker] = data["Close"]
                    time.sleep(2)
                else:
                    st.error(f"Gagal mengunduh data untuk {ticker}: Data is None or Empty")
                    return None
            except Exception as e:
                st.error(f"Gagal mengunduh data untuk {ticker}: {e}")
                return None  # Important: Return None on error to prevent further issues
        return df.dropna(how="all")  # Drop rows where all values are NaN

    df_close = get_data()

    if df_close is None:
        st.stop()  # Stop if data download failed

    # Preprocessing
    scaler = MinMaxScaler()
    try:
        scaled_data = scaler.fit_transform(df_close)
    except ValueError as e:
        st.error(f"ValueError during scaling: {e}")
        st.stop()

    # Sequence preparation
    def create_sequences(data, time_steps=50):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i : i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 50
    X, y = create_sequences(scaled_data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], len(tickers)))

    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build model
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(time_steps, len(tickers))),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(len(tickers)),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    with st.spinner("🚀 Melatih model..."):
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict future
    future_input = scaled_data[-time_steps:].reshape(
        (1, time_steps, len(tickers))
    )  # Reshape for single prediction
    future_predictions = []
    for _ in range(predict_days):
        pred = model.predict(future_input, verbose=0)
        future_predictions.append(pred[0])
        future_input = np.concatenate(
            [future_input[:, 1:, :], pred.reshape((1, 1, len(tickers)))], axis=1
        )

    future_predictions = scaler.inverse_transform(np.array(future_predictions))
    future_df = pd.DataFrame(future_predictions, columns=tickers)
    future_dates = pd.date_range(
        start=df_close.index[-1] + pd.Timedelta(days=1), periods=predict_days, freq="B"
    )

    # Combine actual + predicted
    combined_df = pd.concat(
        [
            df_close[[selected_ticker]].iloc[-100:],
            pd.DataFrame(
                {selected_ticker: future_df[selected_ticker].values}, index=future_dates
            ),
        ]
    )

    # Plot using Plotly
    st.subheader(f"📊 Grafik Harga dan Prediksi ({stock_choice})")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined_df.index[-100:],
            y=combined_df[selected_ticker].iloc[-100:],
            mode="lines",
            name="Actual Price",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_df[selected_ticker],
            mode="lines",
            name="Predicted Price",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Price",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Prediction change %
    start_price = combined_df[selected_ticker].iloc[-predict_days - 1]
    end_price = combined_df[selected_ticker].iloc[-1]
    pct_change = ((end_price - start_price) / start_price) * 100
    st.metric("📈 Persentase Prediksi Kenaikan", f"{pct_change:.2f} %")

    # Function to scrape sentiment data
    def scrape_sentiment(ticker):
        sources = {
            "Twitter": f"https://twitter.com/search?q=${ticker}&src=recent_tweets",
            "Google News": f"https://news.google.com/search?q={ticker}&hl=en-US&gl=US&ceid=US:en",
        }
        sentiment_data = []
        scraped_data = {"Twitter": [], "Google News": []}  # Store the text
        for source, url in sources.items():
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                if source == "Twitter":
                    # Example extraction (adapt to Twitter's current HTML structure)
                    tweet_elements = soup.find_all(
                        "div", attrs={"data-testid": "tweet"}
                    )  # Adapt as needed.
                    tweets = [el.text.strip() for el in tweet_elements]
                    sentiments = [get_sentiment(tweet) for tweet in tweets[:2]]
                    sentiment_data.extend(sentiments)
                    scraped_data["Twitter"].extend(tweets[:2])  # Store
                elif source == "Google News":
                    # Example extraction (adapt to Google News's HTML structure)
                    article_elements = soup.find_all("article")
                    headlines = [el.text.strip() for el in article_elements]
                    sentiments = [get_sentiment(headline) for headline in headlines[:2]]
                    sentiment_data.extend(sentiments)
                    scraped_data["Google News"].extend(headlines[:2])
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data from {source} for {ticker}: {e}")
                sentiment_data.extend(["Netral"] * 2)
                scraped_data[source] = ["N/A"] * 2
            except Exception as e:
                st.error(f"Error processing data from {source} for {ticker}: {e}")
                sentiment_data.extend(["Netral"] * 2)
                scraped_data[source] = ["N/A"] * 2
        return sentiment_data[:7], scraped_data  # return also scraped data

    def get_sentiment(text):
        # Placeholder sentiment analysis function. Replace with a real model.
        #  Here's how you might use TextBlob (install with pip install textblob):
        try:
            from textblob import TextBlob

            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0.1:
                return "Positif"
            elif analysis.sentiment.polarity < -0.1:
                return "Negatif"
            else:
                return "Netral"
        except ImportError:
            return "Netral"
        #  OR
        # You could use Vader (install with pip install vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        # analyzer = SentimentIntensityAnalyzer()
        # vs = analyzer.polarity_scores(text)
        # if vs['compound'] >= 0.05:
        #     return "Positif"
        # elif vs['compound'] <= -0.05:
        #     return "Negatif"
        # else:
        #     return "Netral"
        return "Netral"  # default

    # Scrape and display sentiment
    st.subheader("🧾 Sentimen 7 Hari Terakhir")
    sentiment_data, scraped_data = scrape_sentiment(
        selected_ticker
    )  # Get the scraped data
    sentiment_df = pd.DataFrame(
        {
            "Tanggal": pd.date_range(end=date.today(), periods=len(sentiment_data), freq="B"),
            "Sentimen": sentiment_data,
            "Sumber": ["Twitter", "Twitter", "Google News", "Google News"][
                : len(sentiment_data)
            ],
        }
    )
    st.dataframe(sentiment_df, use_container_width=True)

    # Show the scraped tweets/news
    st.subheader("Scraped Data")  # Change here
    if scraped_data["Twitter"]:
        st.write("**Tweets:**")
        for tweet in scraped_data["Twitter"]:
            st.write(f"- {tweet}")
    else:
        st.write("**Tweets:** N/A")  # added
    if scraped_data["Google News"]:
        st.write("**News Headlines:**")
        for headline in scraped_data["Google News"]:
            st.write(f"- {headline}")
    else:
        st.write("**News Headlines:** N/A")  # added

    # Recommendation
    st.subheader("✅ Rekomendasi")
    if pct_change > 5:
        st.success("Rekomendasi: **BUY** - Prediksi naik signifikan 📈")
    elif pct_change < -5:
        st.error("Rekomendasi: **SELL** - Prediksi turun signifikan 📉")
    else:
        st.warning("Rekomendasi: **HOLD** - Pergerakan stabil 🔄")


elif page == "UI/UX Design Projects":  # Corrected page name
    st.header("UI/UX Design Portfolio")
    st.write("Here are some of my UI/UX design projects and case studies:")

    st.subheader("MEDICARE Mobile App Prototype")
    st.write("An interactive prototype for a healthcare mobile application.")

    figma_embed_code = """
    <iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="800" height="450" src="https://embed.figma.com/proto/VgM0omLsJZo9DbRRrA6zWy/MEDICARE?node-id=1-3321&starting-point-node-id=1%3A3359&embed-host=share" allowfullscreen></iframe>"""
    components.html(figma_embed_code, height=500)  # Adjust height as needed

    # Add descriptions and potentially images related to the design process
    st.write(
        "This project involved user research, wireframing, and high-fidelity prototyping..."
    )
    # st.image("path/to/your/medicare_design_image.png", caption="Medicare App Design")

    # ... other UI/UX projects ...

elif page == "About Me": 
    st.header("Undergraduate Student in data Science")
    st.write(
        "I am an undergraduate student specializing in data science, with a keen interest in data visualization and UI/UX design. My goal is to leverage data to create impactful solutions."
    )
    st.write(
        "I am passionate about transforming complex data into meaningful insights and user-friendly designs. I believe in the power of data to drive decision-making and enhance user experiences."
    )
    st.write(
        "In my free time, I enjoy exploring new data visualization techniques, learning about UI/UX design principles, and working on personal projects that challenge my skills."
    )
    st.write(
        "Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/mushthafa/) or check out my [GitHub](your_github_url) for more of my work."
    )
    st.markdown("---")
    st.subheader("My Skills")
    st.write(
        "I have experience in the following areas:"
    )
    st.write(
        "- Data Visualization: Python (Matplotlib, Seaborn, Plotly), Tableau\n"
        "- Data Analysis: Python (Pandas, NumPy), SQL\n"
        "- Machine Learning: TensorFlow, Keras\n"
        "- UI/UX Design: Figma, Adobe XD\n"
        "- Web Development: HTML, CSS, JavaScript\n"
    )
    st.write(
        "I am always eager to learn new technologies and improve my skills. I believe in continuous learning and staying updated with the latest trends in data science and design."
    )
    st.markdown("---")
    st.subheader("My Education")
    st.write(
        "I am currently pursuing a degree in Data Science at [Your University Name]. I have completed coursework in data analysis, machine learning, and UI/UX design."
    )
    st.write(
        "I have also participated in various workshops and online courses to enhance my skills in data visualization and design."
    )
    st.markdown("---")
    st.subheader("My Hobbies")
    st.write(
        "In my free time, I enjoy exploring new data visualization techniques, learning about UI/UX design principles, and working on personal projects that challenge my skills."
    )
    st.write(
        "- Reading books on data science and design\n"
        "- Participating in hackathons and coding challenges\n"
        "- Exploring new technologies and tools\n"
        "- Traveling and experiencing different cultures\n"
    )
    st.markdown("---")
    st.subheader("My Projects")
    st.write(
        "I have worked on various projects in data visualization, data analysis, and UI/UX design. Here are some of my notable projects:"
    )
    st.write(
        "- [Stock Price Prediction Dashboard](#stock-prediction-project)  \n"
        "- [Data Visualization Projects](#data-visualization-projects)  \n"
        "- [UI/UX Design Projects](#ui-ux-design-projects)  \n"
    )
    st.markdown("---")
    st.subheader("My Interests")
    st.write(
        "I am particularly interested in the following areas of data science and design:"
    )
    st.write(
        "- Data Visualization: Creating interactive and informative visualizations to communicate insights effectively.\n"
        "- UI/UX Design: Designing user-friendly interfaces that enhance user experience and engagement.\n"
        "- Machine Learning: Applying machine learning techniques to solve real-world problems.\n"
        "- Data Storytelling: Using data to tell compelling stories that resonate with audiences.\n"
    )
    st.markdown("---")
    st.subheader("My Future Goals")
    st.write(
        "I aim to further develop my skills in data science and design, and to work on projects that have a positive impact on society. I am also interested in exploring opportunities in data visualization and UI/UX design."
    )
    st.write(
        "I believe that data has the power to transform industries and improve lives, and I am excited to be a part of this journey."
    )
    st.markdown("---")
        
elif page == "Contact":
    st.header("Contact Me")
    st.write(
        "Feel free to reach out to me for collaboration, inquiries, or just to connect!"
    )
    st.write("Email:mushthafa.a.r@gmail.com")
    st.write(
        "- Email: [mushthafa.a.r@gmail.com](mailto:no)  \n"
        "- LinkedIn: [Mushthafa Aminur Rahman](https://www.linkedin.com/in/mushthafa/)  \n"
        "- GitHub: [your_github_url](your_github_url)  \n"
        "- Twitter: [@your_twitter_handle](https://twitter.com/your_twitter_handle)  \n"
        "- Instagram: [@your_instagram_handle](https://instagram.com/your_instagram_handle)  \n"
        "- Facebook: [Mushthafa Aminur Rahman](https://www.facebook.com/your_facebook_profile)  \n"
        "- Portfolio: [your_portfolio_url](your_portfolio_url)  \n"
    )