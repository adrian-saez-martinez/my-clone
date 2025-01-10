__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st

st.set_page_config(page_title="Adrian Saez Martinez", layout="wide")

resume = st.Page("pages/1_resume.py", title="Resume", icon=":material/home:")
conctact_info = st.Page("pages/2_contact.py", title="Contact", icon=":material/info:")
chatbot = st.Page("pages/3_chatbot.py", title="Ask me anything about Adrián", icon=":material/smart_toy:")
#chatbot = st.Page("pages/3_chatbot_langGraph.py", title="Ask me anything about Adrián", icon=":material/smart_toy:")
how_it_works = st.Page("pages/4_how_it_works.py", title="How was the AI Assistant built?", icon=":material/bug_report:")

pg = st.navigation(
    {
        "AI Assistant": [chatbot],
        "How It works": [how_it_works],
        "About me": [resume, conctact_info],
    }
)

pg.run()