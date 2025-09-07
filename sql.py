# """
# Clinical Trial SQL Chatbot using LangGraph SQL Agent
# A Streamlit chatbot that uses LangGraph's SQL agent to query clinical trial database.
# """

# import os
# import sqlite3
# import streamlit as st
# from typing import Dict, Any, List
# from dotenv import load_dotenv

# # LangChain imports
# from langchain_openai import ChatOpenAI
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver

# # Load environment variables
# load_dotenv()

# # Streamlit configuration
# st.set_page_config(
#     page_title="Clinical Trial AI Assistant",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .database-info {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#         margin-bottom: 1rem;
#     }
#     .example-queries {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #28a745;
#     }
# </style>
# """, unsafe_allow_html=True)

# class ClinicalTrialChatbot:
#     """Clinical trial chatbot using SQL agent."""
    
#     def __init__(self):
#         self.db_path = "clinical_trials.db"
#         self.db = None
#         self.agent = None
#         self.memory = MemorySaver()
#         self.initialize_components()
    
#     def check_database_exists(self) -> bool:
#         """Check if the clinical trials database exists."""
#         if not os.path.exists(self.db_path):
#             st.error(f"❌ Database file '{self.db_path}' not found in the current directory.")
#             st.write("Please make sure you have:")
#             st.write("1. Run the CSV to SQL converter script")
#             st.write("2. The `clinical_trials.db` file is in the same folder as this app")
#             return False
#         return True
    
#     def initialize_components(self):
#         """Initialize database connection and SQL agent."""
#         try:
#             # Check API key
#             if not os.getenv("OPENAI_API_KEY"):
#                 st.error("❌ OPENAI_API_KEY not found. Please set it in your environment variables.")
#                 st.stop()
            
#             # Check database
#             if not self.check_database_exists():
#                 st.stop()
            
#             # Initialize database connection
#             self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
#             # Initialize LLM
#             llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
            
#             # Create toolkit with all SQL tools
#             toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
#             tools = toolkit.get_tools()
            
#             # Create system prompt for clinical trial domain
#             system_prompt = self.get_clinical_trial_system_prompt()
            
#             # Create agent with memory
#             self.agent = create_react_agent(
#                 llm, 
#                 tools, 
#                 prompt=system_prompt,
#                 checkpointer=self.memory,
#                 verbose = True
#             )
            
#             st.success("✅ Clinical Trial SQL Agent initialized successfully!")
            
#         except Exception as e:
#             st.error(f"❌ Error initializing components: {e}")
#             st.stop()
    
#     def get_clinical_trial_system_prompt(self) -> str:
#         """Get specialized system prompt for clinical trial data."""
#         return """
# You are a specialized AI assistant for analyzing clinical trial data using SQL queries.
# You have access to a comprehensive clinical trial database with the following structure:

# DATABASE TABLES:
# 1. **trials** - Core trial information (Product/Regimen Name, Trial Acronym/ID, Phase, etc.)
# 2. **efficacy** - Efficacy outcomes (ORR, mPFS, mOS, survival rates, etc.)
# 3. **safety** - Safety data (adverse events, TRAEs, treatment-related deaths, etc.)
# 4. **enrollment** - Patient enrollment data (total enrolled, follow-up periods, etc.)
# 5. **commercial** - Commercial/analysis data (sales, analyst comments, etc.)
# 6. **complete_data** - Complete backup of all original data
# 7. **trial_summary** VIEW - Combined key metrics across tables
# 8. **efficacy_summary** VIEW - Clean efficacy metrics
# 9. **safety_summary** VIEW - Clean safety data

# CLINICAL TRIAL DATA STRUCTURE UNDERSTANDING:
# - Each row represents ONE ARM of a clinical trial
# - **Product/Regimen Name** = Treatment arm being tested
# - **Comparator** = Control/comparison arm (standard of care or placebo)
# - **Trial Acronym/ID** = Unique trial identifier (same for all arms in a trial)
# - **Active Developers (Companies Names)** = Pharmaceutical companies developing the treatment
# - **Highest Phase** = Development status of the treatment
# - When searching for specific treatments in trials, check BOTH Product/Regimen Name AND Comparator columns
# IMPORTANT KEEP IN MIND: Never give hallucination answer if you are confident about metrics values if values are not present then mention as per database value.
# CLINICAL OUTCOMES DEFINITIONS:

# **EFFICACY OUTCOMES:**
# - **ORR** = Overall Response Rate (percentage of patients with tumor shrinkage)
# - **CR** = Complete Response (complete disappearance of tumor)
# - **PR** = Partial Response (significant tumor shrinkage)
# - **mPFS** = Median Progression-Free Survival (time until disease progression)
# - **mOS** = Median Overall Survival (time until death)
# - **mDoR** = Median Duration of Response (how long responses last)
# - **DCR** = Disease Control Rate (patients with stable or shrinking disease)

# **SAFETY OUTCOMES:**
# - **Gr ≥3 TRAEs** = Grade 3 and above Treatment-Related Adverse Events
# - **Gr ≥3 TEAEs** = Grade 3 and above Treatment-Emergent Adverse Events  
# - **Gr ≥3 irAEs** = Grade 3 and above immune-related Adverse Events
# - **Gr ≥3 AEs** = Grade 3 and above Adverse Events
# - **Gr 3/4 TRAEs** = Grade 3 or 4 Treatment-Related Adverse Events
# - **Gr 3/4 TEAEs** = Grade 3 or 4 Treatment-Emergent Adverse Events
# - **Gr 3/4 irAEs** = Grade 3 or 4 immune-related Adverse Events
# - **Gr 3/4 AEs** = Grade 3 or 4 Adverse Events

# TERMINOLOGY VARIATIONS TO RECOGNIZE:

# **Phase Terminology:**
# - Ph1 = Phase 1 = Phase I = Phase1
# - Ph2 = Phase 2 = Phase II = Phase2  
# - Ph3 = Phase 3 = Phase III = Phase3
# - Ph1/2 = Phase 1/2 = Phase I/II

# **Safety Event Variations (all refer to Gr ≥3 TRAEs):**
# - Gr 3+ TRAEs
# - Grade 3 and above TRAEs
# - High grade TRAEs
# - Gr ≥3 treatment related adverse events
# - Gr ≥3 treatment related AEs
# - Gr ≥3 treatment-related adverse events
# - Gr ≥3 treatment-related AEs
# - Gr 3+ treatment related adverse events
# - Gr 3+ treatment related AEs
# - Gr 3+ treatment-related adverse events
# - Gr 3+ treatment-related AEs
# - Grade 3 and above treatment related adverse events
# - Grade 3 and above treatment related AEs
# - Grade 3 and above treatment-related adverse events
# - Grade 3 and above treatment-related AEs
# - High grade TRAEs
# - High grade treatment-related AEs
# - High grade treatment-related adverse events

# **Company Name Variations:**
# - Merck & Co = MSD = Merck/MSD = Merck US = US-based Merck
# - BMS = Bristol Myers Squibb = Bristol-Myers Squibb
# - Roche = Genentech = Roche/Genentech

# **Drug Name Expansions:**
# - Nivolumab = nivo = opdivo = BMS-936558 = ONO-4538
# - Ipilimumab = ipi = yervoy = BMS-734016 = MDX-010
# - Pembrolizumab = pembro = keytruda = MK-3475
# - Atezolizumab = atezo = tecentriq = MPDL3280A
# - Vemurafenib = vemu = zelboraf = PLX4032
# - Dabrafenib = dabra = tafinlar = GSK2118436
# - Trametinib = trame = mekinist = GSK1120212

# DEFAULT TABLE PARAMETERS:
# When presenting results in tables, always include these core columns when available:
# 1. **Regimen name** (Product/Regimen Name)
# 2. **Active developers** (Active Developers Companies Names) 
# 3. **Highest Phase**
# 4. **Line of therapy (LoT)** (from Additional TA Details or Therapeutic Indication)
# 5. **Patient population** (from Biomarkers, Tumor Histology, or Therapeutic Indication)
# Plus relevant outcome metrics based on the query

# DATA FORMAT NOTES:
# - Values stored as TEXT preserving original format ("58.3%", "11.5", "NR")
# - "NR" or "Not reached" = positive outcome for survival endpoints
# - Use LIKE operators for flexible text matching
# - Use CAST for numeric operations when needed

# QUERY APPROACH:
# 1. **Always start by examining table structure** using available tools
# 2. **Search comprehensively across relevant columns** for all terminology variations
# 3. **Use JOINs** to combine data across tables (JOIN ON ID)
# 4. **Include flexible text matching** with multiple LIKE conditions for variations
# 5. **Order results meaningfully** (by efficacy, phase, company, etc.)
# 6. **Limit results appropriately** (10-20 unless specified otherwise)
# 7. **Double-check queries** before execution
# 8. **Present results in clear tabular format** with default parameters

# SEARCH STRATEGY FOR ANY QUERY:
# - Identify key terms and map to database columns
# - Include all possible variations of terminology
# - Search both treatment arms (Product/Regimen Name) and comparator arms (Comparator)
# - Join relevant tables for comprehensive results
# - Apply appropriate filters and sorting
# - Present results with clinical context and interpretation
# - 
# RESPONSE FORMAT:
# 1. Understand the question and identify relevant clinical concepts
# 2. Query systematically using comprehensive terminology matching
# 3. Present results in clear tables with default parameters
# 4. Provide clinical interpretation and context
# 5. Specify data limitations or assumptions when relevant

# Remember: Handle ANY type of clinical trial question flexibly by mapping user terminology to database fields and using comprehensive search strategies across all relevant columns and tables.
# """
    
#     def get_database_info(self) -> Dict[str, Any]:
#         """Get information about the database structure."""
#         try:
#             # Get table names
#             tables = self.db.get_usable_table_names()
            
#             # Get row counts for each table
#             table_info = {}
#             for table in tables:
#                 try:
#                     result = self.db.run(f"SELECT COUNT(*) FROM `{table}`")
#                     count = int(result.strip("[]()").split(",")[0])
#                     table_info[table] = count
#                 except:
#                     table_info[table] = "Unknown"
            
#             return {
#                 "tables": tables,
#                 "table_counts": table_info,
#                 "total_tables": len(tables)
#             }
            
#         except Exception as e:
#             st.error(f"Error getting database info: {e}")
#             return {"tables": [], "table_counts": {}, "total_tables": 0}
    
#     def query_agent(self, question: str, thread_id: str = "default") -> str:
#         """Query the SQL agent with a question."""
#         try:
#             config = {"configurable": {"thread_id": thread_id}}
            
#             # Stream the agent's response
#             messages = []
#             for step in self.agent.stream(
#                 {"messages": [{"role": "user", "content": question}]},
#                 config,
#                 stream_mode="values"
#             ):
#                 messages = step["messages"]
            
#             # Get the final AI message
#             if messages and len(messages) > 0:
#                 final_message = messages[-1]
#                 if hasattr(final_message, 'content'):
#                     return final_message.content
#                 else:
#                     return str(final_message)
#             else:
#                 return "No response generated."
                
#         except Exception as e:
#             return f"Error processing query: {str(e)}"

# def render_sidebar():
#     """Render the sidebar with database info and examples."""
#     with st.sidebar:
#         st.header("🗄️ Database Information")
        
#         if 'chatbot' in st.session_state:
#             db_info = st.session_state.chatbot.get_database_info()
            
#             with st.container():
#                 st.markdown('<div class="database-info">', unsafe_allow_html=True)
#                 st.write(f"**📊 Total Tables:** {db_info['total_tables']}")
#                 st.write("**📋 Table Overview:**")
#                 for table, count in db_info['table_counts'].items():
#                     st.write(f"• {table}: {count} records")
#                 st.markdown('</div>', unsafe_allow_html=True)
        
#         st.header("💡 Example Queries")
#         with st.container():
#             st.markdown('<div class="example-queries">', unsafe_allow_html=True)
            
#             example_queries = [
#                 "What are the top 5 trials with the highest ORR?",
#                 "Compare nivolumab vs pembrolizumab response rates",
#                 "Show me phase 3 trials with PFS data",
#                 "Which treatments have the best safety profile?",
#                 "Find trials with median OS over 20 months",
#                 "What's the average ORR by trial phase?",
#                 "Show checkmate trials and their outcomes",
#                 "Which drugs have the most Grade 3/4 adverse events?",
#                 "Find combination therapies vs monotherapies",
#                 "What trials are approved in the US?"
#             ]
            
#             for i, query in enumerate(example_queries, 1):
#                 if st.button(f"📝 {query}", key=f"example_{i}", use_container_width=True):
#                     st.session_state.current_query = query
#                     st.rerun()
            
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         st.header("ℹ️ Tips")
#         st.markdown("""
#         **Query Tips:**
#         - Use drug names or abbreviations (nivo, pembro, ipi)
#         - Ask about specific metrics (ORR, PFS, OS, safety)
#         - Compare treatments or phases
#         - Ask for summaries or trends
#         - Request specific trial data
        
#         **Data Notes:**
#         - ORR = Overall Response Rate (%)
#         - mPFS = Median Progression-Free Survival  
#         - mOS = Median Overall Survival
#         - NR = Not Reached (positive outcome)
#         """)

# def main():
#     """Main application function."""
    
#     # Header
#     st.markdown('<h1 class="main-header">🔬 Clinical Trial SQL Assistant</h1>', unsafe_allow_html=True)
#     st.markdown("Ask questions about clinical trial data using natural language. The AI will generate and execute SQL queries to find answers.")
    
#     # Initialize chatbot
#     if 'chatbot' not in st.session_state:
#         with st.spinner("Initializing Clinical Trial SQL Agent..."):
#             st.session_state.chatbot = ClinicalTrialChatbot()
    
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
    
#     if 'current_query' not in st.session_state:
#         st.session_state.current_query = None
    
#     # Render sidebar
#     render_sidebar()
    
#     # Main chat interface
#     st.subheader("💬 Chat with Your Clinical Trial Database")
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Handle example query selection
#     if st.session_state.current_query:
#         query = st.session_state.current_query
#         st.session_state.current_query = None  # Clear it
        
#         # Add to messages and process
#         st.session_state.messages.append({"role": "user", "content": query})
        
#         with st.chat_message("user"):
#             st.markdown(query)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing clinical trial data..."):
#                 response = st.session_state.chatbot.query_agent(query)
#                 st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.rerun()
    
#     # Chat input
#     if prompt := st.chat_input("Ask about clinical trials..."):
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing clinical trial data..."):
#                 response = st.session_state.chatbot.query_agent(prompt)
#                 st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
    
#     # Footer
#     with st.expander("🔧 Advanced Options"):
#         st.write("**Thread Management:**")
#         if st.button("🔄 Clear Conversation History"):
#             st.session_state.messages = []
#             st.rerun()
        
#         st.write("**Database Actions:**")
#         if st.button("📊 Show Database Schema"):
#             with st.spinner("Loading database schema..."):
#                 try:
#                     schema_info = st.session_state.chatbot.db.get_table_info()
#                     st.text_area("Database Schema", schema_info, height=300)
#                 except Exception as e:
#                     st.error(f"Error loading schema: {e}")

# if __name__ == "__main__":
#     main()


"""
Clinical Trial SQL Chatbot using LangGraph SQL Agent
A clean Streamlit chatbot interface that uses LangGraph's SQL agent to query clinical trial database.
"""

import os
import sqlite3
import streamlit as st
from typing import Dict, Any, List
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="Clinical Trial AI Assistant",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    /* Hide sidebar */
    .css-1d391kg {
        display: none;
    }
    /* Clean chat interface */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* Clean input box */
    .stChatInput > div {
        border-radius: 25px;
    }
</style>
""", unsafe_allow_html=True)

class ClinicalTrialChatbot:
    """Clinical trial chatbot using SQL agent."""
    
    def __init__(self):
        self.db_path = "clinical_trials.db"
        self.db = None
        self.agent = None
        self.memory = MemorySaver()
        self.initialize_components()
    
    def check_database_exists(self) -> bool:
        """Check if the clinical trials database exists."""
        if not os.path.exists(self.db_path):
            st.error(f"❌ Database file '{self.db_path}' not found in the current directory.")
            st.write("Please make sure you have:")
            st.write("1. Run the CSV to SQL converter script")
            st.write("2. The `clinical_trials.db` file is in the same folder as this app")
            return False
        return True
    
    def initialize_components(self):
        """Initialize database connection and SQL agent."""
        try:
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OPENAI_API_KEY not found. Please set it in your environment variables.")
                st.stop()
            
            # Check database
            if not self.check_database_exists():
                st.stop()
            
            # Initialize database connection
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
            # Initialize LLM
            # llm = ChatOpenAI(model="gpt-5")
            llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)
            # Create toolkit with all SQL tools
            toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
            tools = toolkit.get_tools()
            
            # Create system prompt for clinical trial domain
            system_prompt = self.get_clinical_trial_system_prompt()
            
            # Create agent with memory
            self.agent = create_react_agent(
                llm, 
                tools, 
                prompt=system_prompt,
                checkpointer=self.memory,
                verbose=True
            )
            
            st.success("✅ Clinical Trial SQL Agent initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing components: {e}")
            st.stop()
    
    def get_clinical_trial_system_prompt(self) -> str:
        """Get specialized system prompt for clinical trial data."""
        return """
You are a specialized AI assistant for analyzing clinical trial data using SQL queries.
You have access to a comprehensive clinical trial database with the following structure:
DATABASE TABLES:
1. **trials** - Core trial information (Product/Regimen Name, Comparator, Trial Acronym/ID, Regimen MoAs, Product/Regimen Target, Active Developers, Highest Phase, Trial Phase, Biomarkers, Tumor Histology, etc.)
2. **efficacy** - Efficacy outcomes (ORR, mPFS, mOS, CR, PR, DCR, mDoR, survival rates, hazard ratios, etc.)
3. **safety** - Safety data (Gr ≥3 TRAEs, Gr 3/4 TRAEs, Gr ≥3 TEAEs, Key AEs, treatment-related deaths, etc.)
4. **enrollment** - Patient enrollment data (N, n, Median Follow-up)
5. **commercial** - Commercial/analysis data (Sales, Analyst Comments, Product Tags, Benchmark status, etc.)
6. **complete_data** - Complete backup containing ALL original columns from CSV
7. **trial_summary** VIEW - Pre-joined key metrics (regimen_name, phase, orr, pfs, os, safety_grade3_4)
8. **efficacy_summary** VIEW - Clean efficacy metrics with standardized column names
9. **safety_summary** VIEW - Clean safety data with standardized column names
10. Distinguish between active and completed phases this would be the "Highest Phase" column:
    10.1 - Active Development: This includes Ph0, Ph1, Ph2, Ph3, and Ph4, along with their various combinations (e.g., Ph1/2, Ph2/3, Ph3b/4).
    10.2 - Completed Development: This primarily refers to the "Approved" status, often with regional specifics like "(US)" or "(EU)".

CRITICAL DATA FORMAT NOTES:
- ALL data stored as TEXT preserving original format ("58.3%", "11.5 months", "NR", "Not reached")
- Use LIKE operators for text matching, not exact equals
- Join tables using "ID" column: JOIN efficacy e ON t."ID" = e."ID"
- Use DISTINCT to avoid duplicate results from denormalized structure
- Empty strings ('') represent missing data, not NULL values

CLINICAL TRIAL DATA STRUCTURE:
- Each row = ONE ARM of a clinical trial
- **Product/Regimen Name** = Treatment being tested
- **Comparator** = Control arm (standard care/placebo)
- **Trial Acronym/ID** = Unique trial identifier (same for all arms)
- Multiple arms of same trial share Trial Acronym/ID but have different treatments
- Always search BOTH "Product/Regimen Name" AND "Comparator" columns for comprehensive results


CLINICAL OUTCOMES DEFINITIONS:

**EFFICACY OUTCOMES:**
- **ORR** = Overall Response Rate (percentage of patients with tumor shrinkage)
- **CR** = Complete Response (complete disappearance of tumor)
- **PR** = Partial Response (significant tumor shrinkage)
- **mPFS** = Median Progression-Free Survival (time until disease progression)
- **mOS** = Median Overall Survival (time until death)
- **mDoR** = Median Duration of Response (how long responses last)
- **DCR** = Disease Control Rate (patients with stable or shrinking disease)

**SAFETY OUTCOMES:**
- **Gr ≥3 TRAEs** = Grade 3 and above Treatment-Related Adverse Events
- **Gr ≥3 TEAEs** = Grade 3 and above Treatment-Emergent Adverse Events  
- **Gr ≥3 irAEs** = Grade 3 and above immune-related Adverse Events
- **Gr ≥3 AEs** = Grade 3 and above Adverse Events
- **Gr 3/4 TRAEs** = Grade 3 or 4 Treatment-Related Adverse Events
- **Gr 3/4 TEAEs** = Grade 3 or 4 Treatment-Emergent Adverse Events
- **Gr 3/4 irAEs** = Grade 3 or 4 immune-related Adverse Events
- **Gr 3/4 AEs** = Grade 3 or 4 Adverse Events

DEFAULT TABLE PARAMETERS:
When presenting results in tables, always include these core columns when available:
1. **Regimen name** (Product/Regimen Name)
2. **Active developers** (Active Developers Companies Names) 
3. **Highest Phase**
4. **Line of therapy (LoT)** (from Additional TA Details or Therapeutic Indication)
5. **Patient population** (from Biomarkers, Tumor Histology, or Therapeutic Indication)
Plus relevant outcome metrics based on the query

DATA FORMAT NOTES:
- Values stored as TEXT preserving original format ("58.3%", "11.5", "NR")
- "NR" or "Not reached" = positive outcome for survival endpoints
- Use LIKE operators for flexible text matching
- Use CAST for numeric operations when needed

QUERY APPROACH:
1. **Always start by examining table structure** using available tools
2. **Search comprehensively across relevant columns** for all terminology variations
3. **Use JOINs** to combine data across tables (JOIN ON ID)
4. **Do not follow case sensitive search.
4. **Lookup for column unique values for better retrieval.
5. **Include flexible text matching** with multiple LIKE conditions for variations
6. **Order results meaningfully** (by efficacy, phase, company, etc.)
7. **Limit results appropriately** (10-20 unless specified otherwise)
8. **Double-check queries** before execution
9. **Present results in clear text format** with proper formatting

SEARCH STRATEGY FOR ANY QUERY:
- Start with schema inspection** using sql_db_schema tool
- Identify key terms and map to database columns
- Include all possible variations of terminology
- Always Lookup for columns unique values present before.
- Use flexible matching**: WHERE "column" LIKE '%term%' rather than exact matches
- Search both treatment arms (Product/Regimen Name) and comparator arms (Comparator)
- Join relevant tables for comprehensive results
- Apply appropriate filters and sorting
- Present results with clinical context and interpretation
- Provide complete answer based on complete data.

-  ** For Handle Comparator Queries only:**
    *   If you **do not** find the requested regimen in any Product/Regimen Name field, then check if it is mentioned in the Comparator field of any object.
    *   If you find it in a Comparator field, you MUST respond by:
        a. Stating clearly that the dataset **does not contain the specific performance data for the requested arm** (the comparator).
        b. Then, as a helpful alternative, state that you DO have the data for the treatment it was compared against.
        c. Provide the name of the Product/Regimen Name from that row and its corresponding value for the metric the user asked for.
        
RESPONSE FORMAT:
1. Understand the question and identify relevant clinical concepts
2. Query systematically using comprehensive terminology matching
3. Provide clinical interpretation and context
4. Specify data limitations or assumptions when relevant

Remember: Handle ANY type of clinical trial question flexibly by mapping user terminology to database fields and using comprehensive search strategies across all relevant columns and tables.
"""
    
    def query_agent(self, question: str, thread_id: str = "default") -> str:
        """Query the SQL agent with a question."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream the agent's response
            messages = []
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config,
                stream_mode="values"
            ):
                messages = step["messages"]
            
            # Get the final AI message
            if messages and len(messages) > 0:
                final_message = messages[-1]
                if hasattr(final_message, 'content'):
                    return final_message.content
                else:
                    return str(final_message)
            else:
                return "No response generated."
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    """Main application function."""
    
    # Clean header section
    st.markdown('<h1 class="main-header">🔬 Clinical Trial AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about clinical trial data using natural language</p>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing Clinical Trial SQL Agent..."):
            st.session_state.chatbot = ClinicalTrialChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about clinical trials..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical trial data..."):
                response = st.session_state.chatbot.query_agent(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
