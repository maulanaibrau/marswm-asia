import streamlit as st

from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()

show_pages([
    Page("homepage.py", "MARSWM-ASIA", "💻"),

    Page("prediction.py", "Flood Prediction", icon=":ocean:", in_section=False),

    # Study cases
    Section("Study Cases", ":label:"),
    Page("kameda.py", "Kameda, Japan", icon=":flag-jp:", in_section=True),
    Page("xuanthuy.py", "Xuan Thuy, Viet Nam", icon=":flag-vn:", in_section=True),
    Page("subang.py", "Subang, Indonesia", icon=":flag-id:", in_section=True),

    #Page("faq.py", "FAQ", "❔", in_section=False),
    #Page("contact.py", "Contact", icon="📩", in_section=False),
    #Page("about.py", "About", icon="🖼️", in_section=False) 
])

hide_pages(["Thank you"])

st.image("eAsia_logo.png")
st.markdown("### eAsia Joint Research Project 2021-2024")
#st.title("eAsia Project Machine Learning Streamlit Web App")
#st.subheader("LSTM Method for Inland Flood Modelling")

st.markdown("""
Development of Machine Learning and Remote Sensing-based Water Management Platform in Asian Deltas for Sustainable Agriculture.

""")

st.info("Website: [MARSWM-ASIA](https://marswm-asia.net/)")

st.markdown("---")

st.markdown("""
##### Machine Learning Program (LSTM Method) for Inland Flood Prediction

Train your data before making predictions.

""")

if st.button("Step 1 - Numerical Simulation-based Inland Flood Model", type="primary"):
    st.markdown("[Download Manual](https://drive.google.com/drive/folders/1Icf36OPBgzyIdKyzBmq-ucoQzKFg7MGx?usp=sharing)")
if st.button("Step 2 - LSTM-based Inland Flood Model", type="primary"):
    st.markdown("[Download Manual](https://drive.google.com/drive/folders/1ZI9hN3l3nEkLlk6vRvgOtnSaRzPGZ2wl?usp=sharing)")