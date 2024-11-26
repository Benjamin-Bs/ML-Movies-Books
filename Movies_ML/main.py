import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Streamlit-Setup
st.title("Netflix Film Dauer Vorhersage")
st.write("""
Diese App verwendet lineare Regression, um die Dauer von Filmen basierend auf ihrem Titel und Veröffentlichungsjahr vorherzusagen.
""")

# Daten hochladen
uploaded_file = st.file_uploader("Lade eine CSV-Datei mit Film-Daten hoch", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Daten vorverarbeiten
    st.write("### Originaldaten")
    st.dataframe(df.head())

    # Nur relevante Spalten auswählen
    df = df[['title', 'release_year', 'duration']]

    # Entferne fehlende Werte
    df = df.dropna()

    # Bereinigen der 'duration'-Spalte
    df['duration'] = df['duration'].str.extract('(\d+)')
    df['duration'] = df['duration'].astype(float)

    # **Filter hinzufügen**
    st.sidebar.header("Filteroptionen")

    # Filter für Veröffentlichungsjahr
    release_years = df['release_year'].dropna().unique()
    min_year, max_year = int(release_years.min()), int(release_years.max())
    selected_year_range = st.sidebar.slider("Veröffentlichungsjahr", min_year, max_year, (min_year, max_year))
    df = df[(df['release_year'] >= selected_year_range[0]) & (df['release_year'] <= selected_year_range[1])]

    # Filter für Dauer
    if not df.empty:
        min_duration, max_duration = int(df['duration'].min()), int(df['duration'].max())
        selected_duration_range = st.sidebar.slider("Filmdauer (Minuten)", min_duration, max_duration,
                                                    (min_duration, max_duration))
        df = df[(df['duration'] >= selected_duration_range[0]) & (df['duration'] <= selected_duration_range[1])]

    # Filter für Titel-Suche
    title_filter = st.sidebar.text_input("Suche nach Titel", "")
    if title_filter:
        df = df[df['title'].str.contains(title_filter, case=False, na=False)]

    # Gefilterte Daten anzeigen
    st.write("### Gefilterte Daten")
    st.dataframe(df)

    # Überprüfen, ob genügend Daten übrig sind
    if len(df) < 2:
        st.warning("Zu wenige Daten nach der Filterung. Bitte passe die Filter an, um mehr Daten zu laden.")
    else:
        # Text-Vektorisierung für den 'title'
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        X_title = vectorizer.fit_transform(df['title']).toarray()

        # Unabhängige und abhängige Variablen definieren
        X = np.hstack([X_title, df[['release_year']].values])  # Feature: 'title' (numerisch), 'release_year'
        y = df['duration']  # Zielvariable: 'duration'

        # Schritt 4: Daten aufteilen in Trainings- und Testset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Lineares Regressionsmodell erstellen und trainieren
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Vorhersagen treffen
        y_pred = model.predict(X_test)

        # Modellbewertung
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Modellbewertung")
        st.write(f"- Mittlerer quadratischer Fehler (MSE): {mse:.2f}")
        st.write(f"- Bestimmtheitsmaß (R²): {r2:.2f}")

        # Schritt 8: Visualisierung der Ergebnisse
        st.write("### Visualisierung: Tatsächliche vs. Vorhergesagte Dauer")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', label="Vorhersagen vs. Tatsächliche Werte")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
                label="Ideale Vorhersage")
        ax.set_xlabel("Tatsächliche Dauer (Minuten)")
        ax.set_ylabel("Vorhergesagte Dauer (Minuten)")
        ax.set_title("Lineare Regression: Vorhersage der Dauer eines Films")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Bitte lade eine CSV-Datei hoch, um zu beginnen.")
