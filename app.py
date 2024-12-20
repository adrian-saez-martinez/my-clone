import streamlit as st

st.set_page_config(page_title="Adrian Saez Martinez", layout="wide")

resume = st.Page("pages/1_resume.py", title="Resume", icon=":material/home:")
conctact_info = st.Page("pages/2_contact.py", title="Contact", icon=":material/info:")
chatbot = st.Page("pages/3_chatbot.py", title="Ask me anything about Adrián", icon=":material/smart_toy:")
chatbot_test = st.Page("pages/4_chatbot_test.py", title="TEST_Ask me anything about Adrián", icon=":material/smart_toy:")

pg = st.navigation(
    {
        "About me": [resume, conctact_info],
        "AI Assistant": [chatbot,chatbot_test],
    }
)

pg.run()