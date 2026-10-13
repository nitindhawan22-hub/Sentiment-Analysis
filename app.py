
import streamlit as st
import joblib

# Load saved files
model = joblib.load("SVM.pkl")
tfidf = joblib.load("Tfidf.pkl")
le = joblib.load("Le.pkl")

# App title
st.title("Product Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment.")

# User input
review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        # Convert text into TF-IDF features
        review_tfidf = tfidf.transform([review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Convert encoded label back to text
        sentiment = str(prediction)

        # Display result
        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        elif sentiment == "Negative":
            st.error(f"Sentiment: {sentiment}")
        else:
            st.info(f"Sentiment: {sentiment}")
