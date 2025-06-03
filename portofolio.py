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

# --- NEW IMPORTS for Big Five Personality Prediction ---
import joblib

# --- Basic Page Configuration ---
st.set_page_config(
    page_title="Mushthafa Aminur Rahman - Data Portfolio",
    page_icon=":bar_chart:",  # You can choose a different emoji
    layout="wide",
)

# --- Load the trained model, scaler, and MAE for Big Five (Loaded once at app start) ---
@st.cache_resource
def load_trained_assets():
    try:
        pipeline = joblib.load('bfi_pipeline.pkl')
        scaler = joblib.load('bfi_scaler.pkl')
        bfi_columns = joblib.load('bfi_columns.pkl')
        mae_score = joblib.load('model_mae.pkl')
        return pipeline, scaler, bfi_columns, mae_score
    except FileNotFoundError:
        st.error("Error: Model files (bfi_pipeline.pkl, bfi_scaler.pkl, bfi_columns.pkl, model_mae.pkl) not found.")
        st.error("Please ensure these files are in the same directory as your Streamlit app.")
        st.stop()

# Load assets only once
pipeline, scaler, bfi_columns, mae_score = load_trained_assets()


# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Homepage",
        "Data Visualization Projects",
        "Data Analysis Projects",
        "UI/UX Design Projects",
        "About Me",
        "Contact",
    ],
)

# --- Main Content Area ---
if page == "Homepage":
    # --- Hero Section ---
    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
    with col1:
        # Directly pass the URL to st.image() - Removed try-except for URL
        st.image(
            "https://i.ppy.sh/ded8aaca4957a758f738ff55d8f7d0af53bd55db/68747470733a2f2f6d656469612e74656e6f722e636f6d2f4a3441716e4351304e575541414141432f6f6b6179752d6e656b6f6d6174612d6f6b6179752e676966",
            width=200
        )
        # If you prefer a local image:
        # try:
        #     profile_image = Image.open(r"Foto .png")  # Replace with your local image path
        #     st.image(profile_image, width=200)
        # except FileNotFoundError:
        #     st.error("Profile image not found. Please update the path.")

    with col2:
        st.title("Mushthafa Aminur Rahman")
        st.subheader("Data Visualization Designer & Analyst | UI/UX Enthusiast")
        st.write(
            "Welcome to my online portfolio showcasing my work in data visualization, data analysis, and UI/UX design. Explore my projects below!"
        )
        st.markdown(
            "[LinkedIn](https://www.linkedin.com/in/mushthafa/) | [GitHub](https://github.com/umam1n/)"
        )  # Replace with your actual URLs

    st.markdown("---")

    # --- Featured Projects (Optional) ---
    st.subheader("Featured Projects")
    col3, col4 = st.columns(2)
    with col3:
        try:
            project1_image = Image.open(
                r"29.7 cm x 42 cm.png"
            )  # Replace with your image path
            st.image(project1_image, caption="Infographic of Poverty Rate in Indonesia")
        except FileNotFoundError:
            st.error("Project 1 image not found. Please update the path.")
        st.markdown("[View Project 1](#project-1)")  # Create an anchor link later

    with col4:
        try:
            project2_image = Image.open(
                r"Home.png"
            )  # Replace with your image path
            st.image(project2_image, caption="Visualization Nuclear usage and impact")
        except FileNotFoundError:
            st.error("Project 2 image not found. Please update the path.")
        st.markdown("[View Project 2](#project-2)")  # Create an anchor link later

    # Second row: single column (col5)
    col5 = st.columns(1)[0] # Using [0] to unpack the single column from the tuple
    with col5:
        try:
            project3_image = Image.open(
                r"newplot.png"
            )  # Replace with your image path
            st.image(project3_image, caption="Visualization of Stock Price Prediction")
        except FileNotFoundError:
            st.error("Project 3 image not found. Please update the path.")
        st.markdown("[View Project 3](#project-3)")

    st.markdown("---")

elif page == "Data Visualization Projects":
    st.header("Data Visualization Projects")
    st.write(
        "Visualization of data project."
    )
    st.subheader("Infographic of Poverty Rate in Indonesia")
    st.write(
        "This project visualizes the poverty rate in Indonesia using an infographic format. The infographic highlights key statistics and trends related to poverty in the country."
    )
    try:
        project1_image = Image.open(
            r"29.7 cm x 42 cm.png"
        )  # Replace with your image path
        st.image(project1_image, caption="Infographic of Poverty Rate in Indonesia")
    except FileNotFoundError:
        st.error("Image for Infographic of Poverty Rate not found. Please update the path.")

    # Add more data visualization projects here following the same pattern
    st.markdown("---")
    st.subheader("Power BI Dashboard BI Team 4 Report")
    st.write("An interactive Power BI dashboard displaying key business insights and performance metrics.")
    # Power BI Embed Code
    powerbi_embed_code = """
    <iframe title="BI Team 4" width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiZTgyZjA1OTAtYTE2My00MDNjLTg2ZGUtNzgwODdmZjAxYTg1IiwidCI6IjkwYWZmZTBmLWMyYTMtNDEwOC1iYjk4LTZjZWI0ZTk0ZWYxNSIsImMiOjEwfQ%3D%3D&pageName=08a16d8354001fa1d27a" frameborder="0" allowFullScreen="true"></iframe>"""
    components.html(powerbi_embed_code, height=400) # Adjust height as needed for better viewing

    st.write("This dashboard was developed to provide a comprehensive overview of [mention specific area, e.g., sales performance, operational efficiency] for BI Team 4, allowing for drill-down analysis and interactive filtering.")


elif page == "Data Analysis Projects":
    st.header("Data Analysis Projects")

    # --- Project 1: Stock Price Prediction Dashboard ---
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


    # Function to scrape sentiment data (MODIFIED)
    # Using st.cache_data to avoid re-scraping on every rerun
    @st.cache_data(ttl=3600) # Cache for 1 hour
    def scrape_sentiment(ticker):
        # Updated sources to only include Google News and Yahoo Finance News
        sources = {
            "Google News": f"https://news.google.com/search?q={ticker}&hl=en-US&gl=US&ceid=US:en",
            "Yahoo Finance News": f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}",
        }
        all_scraped_data = {"Google News": [], "Yahoo Finance News": []}  # Store the raw text

        for source_name, url in sources.items():
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                soup = BeautifulSoup(response.content, "html.parser")

                headlines = []
                if source_name == "Google News":
                    # Google News Selectors (NEED TO BE UPDATED based on current HTML)
                    # You'll need to manually inspect Google News HTML to find correct selectors
                    # Example of what you might look for (these are *likely* outdated now):
                    for article_tag in soup.find_all('article'):
                        # Common pattern: h3 with a link inside
                        headline_elem = article_tag.find('h3')
                        if headline_elem:
                            headline_link = headline_elem.find('a')
                            if headline_link and headline_link.text:
                                headlines.append(headline_link.text.strip())
                    # Fallback/additional selectors if articles don't catch all
                    # For example, if headlines are just in specific div/span tags
                    for div_tag in soup.find_all('div', class_='some-google-news-headline-class'):
                        a_tag = div_tag.find('a')
                        if a_tag and a_tag.text:
                            headlines.append(a_tag.text.strip())

                elif source_name == "Yahoo Finance News":
                    # Yahoo Finance News Selectors (NEED TO BE UPDATED based on current HTML)
                    # You'll need to manually inspect Yahoo Finance News HTML to find correct selectors
                    # Common patterns: div containing a link, often with class 'Fw(b)' for bold text
                    for div_tag in soup.find_all('div', class_='Ov(h)'): # This was a common outer div
                        a_tag = div_tag.find('a', class_='Fw(b)')
                        if a_tag and a_tag.text:
                            headlines.append(a_tag.text.strip())
                    # Another common pattern if headlines are within specific tags
                    for h3_tag in soup.find_all('h3', class_='Mb(5px)'): # Another common Yahoo headline tag
                        a_tag = h3_tag.find('a')
                        if a_tag and a_tag.text:
                            headlines.append(a_tag.text.strip())

                # Store unique headlines, max 5 per source
                all_scraped_data[source_name].extend(list(set(headlines))[:5])

            except requests.exceptions.RequestException as e:
                st.warning(f"Failed to fetch data from {source_name} for {ticker}: {e}. Skipping this source.")
                all_scraped_data[source_name] = ["Error: Could not retrieve news."]
            except Exception as e:
                st.warning(f"Error parsing data from {source_name} for {ticker}: {e}. Skipping this source.")
                all_scraped_data[source_name] = ["Error: Could not parse news content."]

        return all_scraped_data

    def get_sentiment(text):
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
            st.warning("TextBlob not installed. Sentiment analysis will be 'Netral'. Run `pip install textblob`.")
            return "Netral"
        except Exception as e:
            # Handle potential errors during TextBlob analysis (e.g., non-string input)
            return "Netral" # Default to Neutral if analysis fails

    # Scrape and display sentiment
    st.subheader("🧾 Sentimen Terbaru dari Berita")
    scraped_data_raw = scrape_sentiment(selected_ticker) # Get only the raw scraped data

    sentiment_list = []
    # Using a counter to assign approximate dates for display
    date_counter = 0
    for source_name, headlines in scraped_data_raw.items():
        if headlines: # Only process if there are actual headlines
            for headline in headlines:
                if headline and headline != "Error: Could not retrieve news." and headline != "Error: Could not parse news content.": # Only process valid headlines
                    sentiment_list.append({
                        "Tanggal": date.today() - timedelta(days=date_counter),
                        "Sumber": source_name,
                        "Sentimen": get_sentiment(headline),
                        "Berita/Judul": headline
                    })
                    date_counter += 1
        else:
            # Add a placeholder if no news was found for a source
            sentiment_list.append({
                "Tanggal": date.today() - timedelta(days=date_counter),
                "Sumber": source_name,
                "Sentimen": "Netral", # Default sentiment if no news
                "Berita/Judul": "Tidak ada berita ditemukan."
            })
            date_counter += 1


    if sentiment_list: # Check if sentiment_list is not empty before creating DataFrame
        sentiment_df = pd.DataFrame(sentiment_list)
        # Sort by date for better display
        sentiment_df = sentiment_df.sort_values(by="Tanggal", ascending=False).reset_index(drop=True)
        st.dataframe(sentiment_df, use_container_width=True)
    else:
        st.info("Tidak dapat mengambil berita sentimen saat ini.") # No sentiment data could be retrieved


    # Show the scraped news headlines directly (refined display)
    st.subheader("Detail Berita yang Ditemukan")
    found_any_news = False
    for source_name, headlines in scraped_data_raw.items():
        if headlines and any(h not in ["Error: Could not retrieve news.", "Error: Could not parse news content."] for h in headlines):
            st.write(f"**{source_name}:**")
            for headline in headlines:
                if headline not in ["Error: Could not retrieve news.", "Error: Could not parse news content."]:
                    st.write(f"- {headline}")
            found_any_news = True
        else:
            st.write(f"**{source_name}:** Tidak ada berita ditemukan atau terjadi kesalahan saat mengambil.")

    if not found_any_news:
        st.info("Tidak ada berita yang berhasil diambil dari sumber manapun.")


    # Recommendation
    st.subheader("✅ Rekomendasi")
    if pct_change > 5:
        st.success("Rekomendasi: **BUY** - Prediksi naik signifikan 📈")
    elif pct_change < -5:
        st.error("Rekomendasi: **SELL** - Prediksi turun signifikan 📉")
    else:
        st.warning("Rekomendasi: **HOLD** - Pergerakan stabil 🔄")

    # --- Horizontal Rule to separate projects ---
    st.markdown("---")

    # --- NEW PROJECT: Big Five Personality Prediction Dashboard ---
    st.subheader("Big Five Personality Prediction Dashboard")
    st.markdown(
        """
        This project showcases a machine learning model that predicts Big Five personality traits
        (Openness, Conscientiousness, Extroversion, Agreeableness, Neuroticism) based on text input.
        It uses a pre-trained pipeline to analyze the sentiment and characteristics of the provided text.
        """
    )
    st.write("---")

    # Display MAE in the sidebar for visibility
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric(label="Mean Absolute Error (MAE) on Test Set", value=f"{mae_score:.4f}")
    st.sidebar.markdown(
        """
        <small>MAE indicates the average magnitude of the errors in a set of predictions, without considering their direction. A lower MAE is better.</small>
        """, unsafe_allow_html=True
    )

    # --- Example Sentences ---
    example_sentences = {
        "Select an example...": "", # Default empty selection
        "Positive & Social": "Having a wonderful time with friends, life is full of joy!",
        "Reflective & Creative": "Lost in thought, I often find inspiration in the quiet moments of nature.",
        "Organized & Diligent": "I meticulously plan my tasks to ensure everything is completed on time and efficiently.",
        "Expressing Frustration": "This situation is incredibly frustrating and making me feel very anxious.",
        "Indonesian Phrase Example": "astaga Indonesia bukan negara fatherless lagi karena PAPAH SUDAH DATANG"
    }

    selected_example = st.selectbox("Or choose an example sentence:", list(example_sentences.keys()))

    # Set the text area content based on selection
    user_input_default = example_sentences[selected_example] if selected_example else ""
    user_input = st.text_area("Enter your sentence here:", value=user_input_default, height=100)


    if st.button("Predict Personality"):
        if user_input:
            # Check for keywords and display image
            if "job" in user_input.lower() and "application" in user_input.lower():
                st.subheader("Recognized Keywords: 'job' and 'application'!")
                st.image("https://eforms.com/images/2018/03/Employment-Job-Application.png", caption="Job Application Form Example", use_container_width=True)
                st.markdown("---") # Add a separator after the image

            # Predict personality scores
            prediction_scaled = pipeline.predict([user_input])

            st.subheader("Predicted Big Five Scores (0-1 Scale)")

            results_df = pd.DataFrame(prediction_scaled, columns=bfi_columns)
            results_df_rounded = results_df.round(4)

            st.dataframe(results_df_rounded.style.highlight_max(axis=1))

            st.write("---")
            st.markdown(
                """
                **Understanding the Big Five Personality Traits (Scores 0-1):**
                * **Openness:** Imaginative, insightful, and curious vs. practical, conventional, and conservative.
                * **Conscientiousness:** Organized, thorough, and reliable vs. careless, impulsive, and disorganized.
                * **Extroversion:** Outgoing, energetic, and assertive vs. solitary, reserved, and thoughtful.
                * **Agreeableness:** Friendly, compassionate, and cooperative vs. challenging, detached, and suspicious.
                * **Neuroticism:** Sensitive, nervous, and prone to worry vs. stable, calm, and secure.
                """
            )
        else:
            st.warning("Please enter some text to make a prediction.")

    st.write("---")
    st.caption("This model provides predictions based on a trained dataset and may not be perfectly accurate for all inputs.")


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
    st.header("Undergraduate Student in Data Science")
    st.write(
        "I am an undergraduate student specializing in data science, with a keen interest in data visualization and UI/UX design. My goal is to leverage data to create impactful solutions."
    )
    st.write(
        "I am passionate about transforming complex data into meaningful insights and user-friendly designs. I believe in the power of data to drive decision-making and enhance user experiences."
    )
    st.write(
        "In my free time, I enjoy reading, learning about design principles of art as in 3d and UI/UX, and working on personal projects that challenge my skills."
    )
    st.write(
        "Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/mushthafa/)."
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
        "I am currently pursuing a degree in Data Science at Telkom University Bandung. I have completed coursework in data analysis, machine learning, and UI/UX design."
    )
    st.write(
        "I have also participated in various workshops and online courses to enhance my skills in data visualization and design."
    )
    st.markdown("---")
    st.subheader("Certifications and competencies")
    st.write(
        "Currently I have some certifications that stand for some of my skills."
    )
    st.write(
        "- English Proficiency Test : [EPRT](https://igracias.telkomuniversity.ac.id/LACValidation/index.php?id=2240801&id2=84&id3=2243155&id4=1305220006)  \n"
        "- English Communication Competency Test: [ECCT](https://igracias.telkomuniversity.ac.id/LACValidation/index.php?id=1097127&id2=107&id3=1099212&id4=1305220006)  \n"
    )
    st.markdown("---")
    st.subheader("My Hobbies")
    st.write(
        "In my free time, I enjoy reading, learning about design principles especially in 3d and UI/UX, and working on personal projects that challenge my skills."
    )
    st.write(
        "- Reading books\n"
        "- Making and designing some artwork as in 3d or else\n"
        "- Exploring new technologies and tools\n"
        "- Traveling and experiencing different cultures\n"
    )
    st.markdown("---")
    st.subheader("My Projects")
    st.write(
        "I have worked on various projects in data visualization, data analysis, and UI/UX design. Here are some of my notable projects:"
    )
    st.write(
        "- [Stock Price Prediction Dashboard](#stock-price-prediction-dashboard)  \n" # Updated anchor
        "- [Big Five Personality Prediction Dashboard](#big-five-personality-prediction-dashboard)  \n" # New anchor
        "- [Infographic of Poverty Rate in Indonesia](#infographic-of-poverty-rate-in-indonesia)  \n" # Updated anchor
        "- [Power BI Dashboard BI Team 4 Report](#power-bi-dashboard-bi-team-4-report)  \n" # Updated anchor
        "- [MEDICARE Mobile App Prototype](#medicare-mobile-app-prototype)  \n" # Updated anchor
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
    st.write(
        "- Email: [mushthafa.a.r@gmail.com](mailto:mushthafa.a.r@gmail.com)  \n" # Corrected email link
        "- LinkedIn: [Mushthafa Aminur Rahman](https://www.linkedin.com/in/mushthafa/)  \n"
        "- GitHub: [umam1n](https://github.com/umam1n)  \n"
        "- Instagram: [@faaffa_](https://www.instagram.com/faaffa_/)  \n"
        "- Portfolio: [Streamlit Portofolio](https://portofolio-umamln.streamlit.app/)  \n"
    )
