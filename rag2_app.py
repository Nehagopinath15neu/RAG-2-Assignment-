import streamlit as st
from rag2_backend import get_llm_response  # Adjust as per your backend filename

# Streamlit UI configuration
st.set_page_config(page_title="Enhanced Chatbot with Feedback", page_icon="💬")
st.title("Enhanced RAG-Enabled Chatbot 💬")

# Conversation history storage
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Mode selector for RAG or Non-RAG
mode = st.radio("Choose Query Mode:", ["Non-RAG Mode", "RAG Mode"])
user_query = st.text_input("Enter your query:")
model_name = st.selectbox("Choose Model:", ["llama3", "mistral"])
domain = st.selectbox("Choose Domain (RAG Mode only):", ["None", "Technology", "Health", "Wikipedia"])

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)

# If the user submits a query, call the LLM
if st.button("Send"):
    if user_query:
        use_rag = mode == "RAG Mode"
        response = get_llm_response(model_name, user_query, use_rag=use_rag, domain=domain)
        
        # Store conversation in session state
        st.session_state.conversation_history.append(f"User: {user_query}")
        st.session_state.conversation_history.append(f"Bot: {response}")
        
        # Display the latest response
        st.write(f"Bot: {response}")

# Feedback section
st.subheader("Rate the Response")
rating = st.radio("Rate the quality of this response:", [1, 2, 3, 4, 5], index=4)
feedback_text = st.text_area("Additional feedback (optional):")

if st.button("Submit Feedback"):
    # Save feedback
    st.session_state.conversation_history.append(f"Rating: {rating}/5")
    st.session_state.conversation_history.append(f"Feedback: {feedback_text}")
    st.write("Thank you for your feedback!")

# Show feedback history
st.subheader("Feedback History")
for feedback in st.session_state.conversation_history:
    if "Rating:" in feedback or "Feedback:" in feedback:
        st.write(feedback)

















