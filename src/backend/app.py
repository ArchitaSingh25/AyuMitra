import streamlit as st
from rag_logic import load_db, build_conversational_chain, save_feedback
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import dotenv
import time

dotenv.load_dotenv()
file_path = os.getenv("CSS_PATH")

# ================================
# Load external CSS
# ================================
def load_css(file_path: str):
    with open(file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(file_path)

# ================================
# Embeddings (must match training)
# ================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ================================
# Streamlit UI Configuration
# ================================
st.set_page_config(
    page_title="Chat with your AyuMITRA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown('<div class="font-size">🤖 Ayumitra </div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">This is an ayurvedic chatbot trained under professional ayurvedic doctors to provide personalized health advice.</div>', unsafe_allow_html=True)

    if st.button("Clear chat history", key="clear_history"):
        st.session_state.clear()
        st.rerun()

# ================================
# Header
# ================================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">How can I assist you today?</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Load DB + Conversational Chain
# ================================
db = load_db(embeddings)
conv_chain, retriever, config = build_conversational_chain(db, session_id="chat1")

# ================================
# Initialize session state
# ================================
if "history" not in st.session_state:
    st.session_state["history"] = []   # [(sender, text)]

if "session_feedback" not in st.session_state:
    st.session_state["session_feedback"] = None  # store once per session

# ================================
# Render chat history
# ================================
for sender, msg in st.session_state["history"]:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AyuMITRA:** {msg}")

# ================================
# Doctor Feedback (once per session)
# ================================
if st.session_state["history"]:  # only show after at least one answer
    with st.expander("📝 Doctor Feedback"):
        if st.session_state["session_feedback"]:
            st.success(f"✅ Feedback saved: {st.session_state['session_feedback']}")
        else:
            with st.form("session_feedback_form"):
                fb = st.text_area("Enter your feedback here:", key="session_fb")
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    if fb.strip():
                        # Save last query + last answer
                        user_q = next((msg for sender, msg in reversed(st.session_state["history"]) if sender == "You"), "")
                        bot_ans = next((msg for sender, msg in reversed(st.session_state["history"]) if sender == "Bot"), "")
                        save_feedback(user_q, bot_ans, fb)
                        st.session_state["session_feedback"] = fb
                        st.success("✅ Feedback saved successfully!")
                    else:
                        st.warning("Please provide feedback before submitting.")

# ================================
# Query Input (always at bottom)
# ================================
query = st.text_area(
    "Type your message...",
    placeholder="Ask your question here...",
    height=80,
    key="query_input"
)

if st.button("Send", key="submit_query"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Processing your query..."):
            try:
                response = conv_chain.invoke({"question": query}, config=config)

                # Append user message
                st.session_state["history"].append(("You", query))

                # Typing effect for bot response
                placeholder = st.empty()
                generated_text = ""
                for word in response.split():
                    generated_text += word + " "
                    placeholder.markdown(f"**🤖 AyuMITRA:** {generated_text}")
                    time.sleep(0.05)

                # Save final bot response
                st.session_state["history"].append(("Bot", response))

                # Clear query input
                st.session_state.query_input = ""

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown(
    '<div class="footer">'
    'Powered by VNIT(Nagpur) | Ayurvedic Knowledge Base'
    '</div>',
    unsafe_allow_html=True
)
