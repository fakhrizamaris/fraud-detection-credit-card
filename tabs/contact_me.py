"""
Contact Me Tab - Developer profile and contact information
"""
import streamlit as st


def render():
    """Render tab Contact Me"""
    st.title("Contact Information")
    st.markdown("---")
    
    # Author Section
    st.markdown("## Author")
    
    st.markdown("""
    **[Your Full Name]**  
    *Student / Data Science Enthusiast*
    
    Department of [Your Department]  
    [University Name]  
    [City, Country]
    """)
    
    st.markdown("---")
    
    # Contact Details
    st.markdown("## Contact Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Email Address**  
        📧 your.email@university.edu
        
        **Phone**  
        📱 +62 xxx-xxxx-xxxx
        
        **Location**  
        📍 [City, Province, Indonesia]
        """)
    
    with col2:
        st.markdown("""
        **LinkedIn**  
        💼 [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
        
        **GitHub**  
        🔗 [github.com/yourusername](https://github.com/yourusername)
        
        **ORCID** *(Optional)*  
        🆔 [0000-0000-0000-0000](https://orcid.org/0000-0000-0000-0000)
        """)
    
    st.markdown("---")
    
    # Academic Background
    st.markdown("## Academic Background")
    
    st.markdown("""
    | Degree | Institution | Year |
    |--------|-------------|------|
    | Bachelor of [Field] | [University Name] | 20XX - Present |
    | High School | [School Name] | 20XX |
    """)
    
    st.markdown("---")
    
    # Research Interests
    st.markdown("## Research Interests")
    
    st.markdown("""
    - Machine Learning & Artificial Intelligence
    - Fraud Detection Systems
    - Data Mining & Pattern Recognition
    - Financial Technology (FinTech)
    """)
    
    st.markdown("---")
    
    # About This Project
    st.markdown("## About This Project")
    
    st.markdown("""
    This **Credit Card Fraud Detection System** is developed as part of [course name / thesis / research project] 
    to demonstrate the application of machine learning techniques in financial security.
    
    **Objectives:**
    1. Implement Random Forest classifier for fraud detection
    2. Handle imbalanced dataset using SMOTE technique
    3. Develop interactive web-based dashboard using Streamlit
    4. Provide real-time transaction analysis capability
    
    **Technologies & Tools:**
    - **Programming Language:** Python 3.x
    - **ML Libraries:** Scikit-learn, Imbalanced-learn
    - **Web Framework:** Streamlit
    - **Visualization:** Altair, Matplotlib
    - **Data Processing:** Pandas, NumPy
    """)
    
    st.markdown("---")
    
    # Acknowledgments
    st.markdown("## Acknowledgments")
    
    st.markdown("""
    I would like to express my gratitude to:
    - [Supervisor Name] for guidance and supervision
    - [University/Institution Name] for providing resources
    - The open-source community for excellent tools and libraries
    """)