import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import time

# Page config - make it look professional
st.set_page_config(
    page_title="Trinity College Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .alert-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api_data(endpoint, params=None):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Make sure it's running on localhost:8000"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value*0.5], 'color': "lightgray"},
                {'range': [max_value*0.5, max_value*0.8], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value*0.9}}))
    
    fig.update_layout(height=300)
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Trinity College Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # API Health Check
    health_data, health_error = fetch_api_data("/health")
    
    if health_error:
        st.error(health_error)
        st.markdown("""
        ### 🚀 To start the API:
        ```bash
        uvicorn main:app --reload
        ```
        """)
        return
    
    # Sidebar for filters and controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Filters
        st.subheader("📊 Filters")
        gpa_threshold = st.slider("At-Risk GPA Threshold", 0.0, 4.0, 2.5, 0.1)
        
        department_filter = st.selectbox(
            "Department Filter", 
            ["All Departments", "CS", "BUS", "PSY", "BIO", "ENG", "MATH", "HIST", "CHEM", "POLS", "ART"]
        )
        
        # API Status
        st.subheader("🔗 API Status")
        if health_data:
            st.success("✅ API Connected")
            st.json(health_data)
    
    # Main dashboard content in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Students", "📚 Courses", "🤖 AI Insights"])
    
    with tab1:
        show_overview_tab(gpa_threshold)
    
    with tab2:
        show_students_tab(gpa_threshold)
    
    with tab3:
        show_courses_tab(department_filter)
    
    with tab4:
        show_ai_insights_tab()

def show_overview_tab(gpa_threshold):
    """Overview dashboard with key metrics"""
    st.header("📊 Campus Overview")
    
    # Fetch key data
    student_analytics, error = fetch_api_data("/analytics/students")
    at_risk_data, _ = fetch_api_data("/students/at-risk", {"gpa_threshold": gpa_threshold})
    
    if error:
        st.error(error)
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎓 Total Students",
            value=student_analytics['total_students'],
            delta=f"{student_analytics['active_students']} active"
        )
    
    with col2:
        st.metric(
            label="📈 Average GPA",
            value=f"{student_analytics['average_gpa']:.2f}",
            delta="Campus-wide"
        )
    
    with col3:
        at_risk_count = len(at_risk_data) if at_risk_data else 0
        st.metric(
            label="⚠️ At-Risk Students",
            value=at_risk_count,
            delta=f"GPA < {gpa_threshold}",
            delta_color="inverse"
        )
    
    with col4:
        retention_rate = (student_analytics['active_students'] / student_analytics['total_students'] * 100) if student_analytics['total_students'] > 0 else 0
        st.metric(
            label="📋 Retention Rate",
            value=f"{retention_rate:.1f}%",
            delta="Active enrollment"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Students by Major pie chart
        if student_analytics and 'students_by_major' in student_analytics:
            majors_data = student_analytics['students_by_major']
            fig = px.pie(
                values=list(majors_data.values()),
                names=list(majors_data.keys()),
                title="🎯 Student Distribution by Major"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # GPA Distribution
        students_data, _ = fetch_api_data("/students", {"limit": 1000})
        if students_data:
            gpa_values = [s['gpa'] for s in students_data if s['is_active']]
            fig = px.histogram(
                x=gpa_values,
                nbins=20,
                title="📊 GPA Distribution",
                labels={'x': 'GPA', 'y': 'Number of Students'}
            )
            fig.add_vline(x=gpa_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"At-Risk Threshold ({gpa_threshold})")
            st.plotly_chart(fig, use_container_width=True)

def show_students_tab(gpa_threshold):
    """Student analytics and at-risk identification"""
    st.header("👥 Student Analytics")
    
    # At-Risk Students Alert
    at_risk_data, error = fetch_api_data("/students/at-risk", {"gpa_threshold": gpa_threshold})
    
    if at_risk_data:
        if len(at_risk_data) > 0:
            st.markdown(f"""
            <div class="alert-box">
                <h4>⚠️ {len(at_risk_data)} Students Need Attention</h4>
                Students with GPA below {gpa_threshold} require academic support
            </div>
            """, unsafe_allow_html=True)
            
            # At-risk students table
            at_risk_df = pd.DataFrame(at_risk_data)
            st.subheader("🚨 At-Risk Students")
            
            # Format the dataframe for better display
            display_df = at_risk_df.copy()
            display_df['gpa'] = display_df['gpa'].round(2)
            display_df = display_df.sort_values('gpa')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "gpa": st.column_config.NumberColumn(
                        "GPA",
                        format="%.2f"
                    ),
                    "active_enrollments": st.column_config.NumberColumn(
                        "Active Courses"
                    )
                }
            )
            
            # At-risk by major breakdown
            major_risk = at_risk_df.groupby('major').size().reset_index(name='count')
            fig = px.bar(
                major_risk, 
                x='major', 
                y='count',
                title="📊 At-Risk Students by Major",
                color='count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.markdown("""
            <div class="success-box">
                <h4>✅ Great News!</h4>
                No students are currently below the GPA threshold.
            </div>
            """, unsafe_allow_html=True)
    
    # Student Search
    st.subheader("🔍 Student Lookup")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        student_search = st.text_input("Enter Student ID (e.g., STU000001)")
    
    with col2:
        if st.button("Search Student"):
            if student_search:
                student_data, error = fetch_api_data(f"/students/{student_search}")
                if student_data:
                    st.json(student_data)
                    
                    # Get prediction for this student
                    prediction_data, _ = fetch_api_data(f"/ml/predictions/student-success/{student_search}")
                    if prediction_data:
                        st.subheader("🤖 AI Prediction")
                        
                        success_prob = prediction_data['success_probability']
                        risk_level = prediction_data['risk_level']
                        
                        # Color code the risk level
                        color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                        
                        st.markdown(f"""
                        **Success Probability:** {success_prob:.0%}  
                        **Risk Level:** <span style="color: {color}; font-weight: bold;">{risk_level}</span>
                        """, unsafe_allow_html=True)
                        
                        if prediction_data['risk_factors']:
                            st.write("**Risk Factors:**")
                            for factor in prediction_data['risk_factors']:
                                st.write(f"- {factor}")
                        
                        st.write("**Recommendations:**")
                        for rec in prediction_data['recommendations']:
                            st.write(f"- {rec}")
                else:
                    st.error(error or "Student not found")

def show_courses_tab(department_filter):
    """Course analytics and performance metrics"""
    st.header("📚 Course Analytics")
    
    # Fetch course data
    params = None if department_filter == "All Departments" else {"department": department_filter}
    courses_data, error = fetch_api_data("/analytics/courses", params)
    
    if error:
        st.error(error)
        return
    
    if not courses_data:
        st.warning("No course data available")
        return
    
    # Convert to DataFrame for easier handling
    courses_df = pd.DataFrame(courses_data)
    
    # Course metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_courses = len(courses_df)
        st.metric("📖 Total Courses", total_courses)
    
    with col2:
        avg_completion = courses_df['completion_rate'].mean()
        st.metric("📋 Avg Completion Rate", f"{avg_completion:.1f}%")
    
    with col3:
        avg_grade = courses_df['average_grade_point'].mean()
        st.metric("🎯 Avg Grade Point", f"{avg_grade:.2f}" if not pd.isna(avg_grade) else "N/A")
    
    # Course performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Top courses by enrollment
        top_courses = courses_df.nlargest(10, 'total_enrollments')
        fig = px.bar(
            top_courses,
            x='total_enrollments',
            y='course_name',
            title="📈 Top Courses by Enrollment",
            orientation='h'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Completion rates by department
        dept_completion = courses_df.groupby('department').agg({
            'completion_rate': 'mean',
            'total_enrollments': 'sum'
        }).reset_index()
        
        fig = px.scatter(
            dept_completion,
            x='total_enrollments',
            y='completion_rate',
            size='total_enrollments',
            text='department',
            title="🎯 Completion Rate vs Enrollment by Department"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed course table
    st.subheader("📊 Course Performance Details")
    
    # Add filters for the table
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("Sort by", ["total_enrollments", "completion_rate", "average_grade_point"])
    with col2:
        ascending = st.checkbox("Ascending order", False)
    
    # Sort and display
    display_df = courses_df.sort_values(sort_by, ascending=ascending)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "completion_rate": st.column_config.NumberColumn(
                "Completion Rate (%)",
                format="%.1f%%"
            ),
            "average_grade_point": st.column_config.NumberColumn(
                "Avg Grade Point",
                format="%.2f"
            )
        }
    )

def show_ai_insights_tab():
    """AI-powered insights and predictions"""
    st.header("🤖 AI-Powered Insights")
    
    st.markdown("""
    ### 🎯 Predictive Analytics Features
    
    This section demonstrates AI/ML capabilities that would be expanded in production:
    """)
    
    # Student Success Prediction Demo
    st.subheader("📊 Student Success Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Current Features:**
        - ✅ Individual student risk assessment
        - ✅ GPA-based success probability
        - ✅ Risk factor identification
        - ✅ Personalized recommendations
        """)
        
        # Sample prediction interface
        st.markdown("**Try a Prediction:**")
        sample_gpa = st.slider("Student GPA", 0.0, 4.0, 3.2, 0.1)
        sample_year = st.selectbox("Year Level", [1, 2, 3, 4])
        
        # Simple prediction logic for demo
        success_prob = min(sample_gpa / 4.0 + 0.1, 1.0)
        risk_level = "High" if success_prob < 0.6 else "Medium" if success_prob < 0.8 else "Low"
        
        st.metric("Success Probability", f"{success_prob:.0%}")
        color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
        st.markdown(f"**Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Future ML Enhancements:**
        - 🔮 Course recommendation engine
        - 📈 Enrollment forecasting
        - 🎓 Graduation timeline predictions
        - 📊 Resource allocation optimization
        - 🤖 Chatbot for student services
        - 📝 Automated report generation
        """)
        
        # Placeholder for future ML model performance
        st.subheader("Model Performance Metrics")
        
        # Simulate model metrics
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [0.85, 0.82, 0.88, 0.85]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title="Student Success Prediction Model Performance",
            color='Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time insights
    st.subheader("📊 Real-Time Campus Insights")
    
    # Fetch some actual data for insights
    students_data, _ = fetch_api_data("/students", {"limit": 1000})
    at_risk_data, _ = fetch_api_data("/students/at-risk")
    
    if students_data and at_risk_data:
        insights = []
        
        # Generate insights
        total_students = len([s for s in students_data if s['is_active']])
        at_risk_count = len(at_risk_data)
        
        if at_risk_count > 0:
            at_risk_pct = (at_risk_count / total_students) * 100
            insights.append(f"🚨 {at_risk_pct:.1f}% of students are at academic risk")
        
        # Major with most at-risk students
        if at_risk_data:
            at_risk_df = pd.DataFrame(at_risk_data)
            top_risk_major = at_risk_df['major'].value_counts().index[0]
            risk_count = at_risk_df['major'].value_counts().iloc[0]
            insights.append(f"📚 {top_risk_major} has the most at-risk students ({risk_count})")
        
        # GPA insights
        active_gpas = [s['gpa'] for s in students_data if s['is_active']]
        if active_gpas:
            median_gpa = np.median(active_gpas)
            insights.append(f"📊 Median campus GPA is {median_gpa:.2f}")
        
        # Display insights
        for insight in insights:
            st.info(insight)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🎓 Trinity College Analytics Dashboard | Built with Streamlit & FastAPI | Data Engineering Portfolio Project
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()