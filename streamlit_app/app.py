import streamlit as st
import requests
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Morning Market Brief",
    page_icon="📈",
    layout="wide"
)

st.title("🗣️ Morning Market Brief")

# Add sidebar for settings
st.sidebar.header("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes")
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Health check function
def check_backend_health():
    try:
        res = requests.get("http://localhost:8005/health", timeout=5)
        return res.status_code == 200
    except:
        return False

# Backend status indicator
if st.sidebar.button("🔍 Check Backend Status"):
    if check_backend_health():
        st.sidebar.success("✅ Backend is running")
    else:
        st.sidebar.error("❌ Backend not responding")
        st.sidebar.write("Make sure your backend server at localhost:8005 is running")

@st.cache_data(ttl=300)
def get_market_brief():
    try:
        res = requests.get("http://localhost:8005/brief", timeout=10)
        return res.status_code, res.text, res.json() if res.status_code == 200 else None
    except requests.exceptions.Timeout:
        return None, "Request timed out", None
    except requests.exceptions.ConnectionError:
        return None, "Connection error - is the backend running?", None
    except Exception as e:
        return None, f"Unexpected error: {str(e)}", None

# Display last updated time
col1, col2 = st.columns([3, 1])
with col2:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# === Chat Interface ===
st.markdown("## 💬 Ask a Question")
user_query = st.text_input("Type your question here", placeholder="e.g., What’s our risk exposure in Asia tech stocks today?")

if st.button("📤 Send Question"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                res = requests.post("http://localhost:8005/chat", json={"query": user_query}, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("response", "No response received.")
                    st.markdown("**🧠 Answer:**")
                    st.markdown(f"“{answer}”")
                else:
                    st.error(f"❌ HTTP {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Auto-refresh logic
if auto_refresh:
    st.info("🔄 Auto-refresh enabled - page will update every 5 minutes")
    time.sleep(300)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Morning Market Brief | Powered by Streamlit</small>
    </div>
    """, 
    unsafe_allow_html=True
)
