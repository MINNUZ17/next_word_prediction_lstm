import streamlit as st
from utils import predict_top_k_words

st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("🔮 Next Word Predictor")
st.write("Type a sentence, and click on a suggestion to autocomplete!   Click 'Press Enter to apply' to start suggestion...")
# Input box state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def update_input(word):
    st.session_state.input_text += " " + word

# Input box
st.text_input("Enter a sentence:", key="input_text")

# Show suggestions
if st.session_state.input_text.strip():
    suggestions = predict_top_k_words(st.session_state.input_text)

    st.markdown("### 🔁 Suggestions:")
    for word in suggestions:
        st.button(word, on_click=update_input, args=(word,))

    full = st.session_state.input_text
    st.markdown(f"**Current sentence:** {full}")
