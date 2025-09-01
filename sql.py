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
            llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            
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
                checkpointer=self.memory
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
1. **trials** - Core trial information (Product/Regimen Name, Trial Acronym/ID, Phase, etc.)
2. **efficacy** - Efficacy outcomes (ORR, mPFS, mOS, survival rates, etc.)
3. **safety** - Safety data (adverse events, TRAEs, treatment-related deaths, etc.)
4. **enrollment** - Patient enrollment data (total enrolled, follow-up periods, etc.)
5. **commercial** - Commercial/analysis data (sales, analyst comments, etc.)
6. **complete_data** - Complete backup of all original data
7. **trial_summary** VIEW - Combined key metrics across tables
8. **efficacy_summary** VIEW - Clean efficacy metrics
9. **safety_summary** VIEW - Clean safety data

CLINICAL TRIAL DATA STRUCTURE UNDERSTANDING:
- Each row represents ONE ARM of a clinical trial
- **Product/Regimen Name** = Treatment arm being tested
- **Comparator** = Control/comparison arm (standard of care or placebo)
- **Trial Acronym/ID** = Unique trial identifier (same for all arms in a trial)
- **Active Developers (Companies Names)** = Pharmaceutical companies developing the treatment
- **Highest Phase** = Development status of the treatment
- When searching for specific treatments in trials, check BOTH Product/Regimen Name AND Comparator columns

IMPORTANT: Never provide hallucinated answers. Only give metric values if you are confident. If the values are not available, explicitly state: 'As per database values'.
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

TERMINOLOGY VARIATIONS TO RECOGNIZE:

**Phase Terminology:**
- Ph1 = Phase 1 = Phase I = Phase1
- Ph2 = Phase 2 = Phase II = Phase2  
- Ph3 = Phase 3 = Phase III = Phase3
- Ph1/2 = Phase 1/2 = Phase I/II

**Safety Event Variations (all refer to Gr ≥3 TRAEs):**
- Gr 3+ TRAEs
- Grade 3 and above TRAEs
- High grade TRAEs
- Gr ≥3 treatment related adverse events
- Gr ≥3 treatment related AEs
- Gr ≥3 treatment-related adverse events
- Gr ≥3 treatment-related AEs
- Gr 3+ treatment related adverse events
- Gr 3+ treatment related AEs
- Gr 3+ treatment-related adverse events
- Gr 3+ treatment-related AEs
- Grade 3 and above treatment related adverse events
- Grade 3 and above treatment related AEs
- Grade 3 and above treatment-related adverse events
- Grade 3 and above treatment-related AEs
- High grade TRAEs
- High grade treatment-related AEs
- High grade treatment-related adverse events

**Company Name Variations:**
- Merck & Co = MSD = Merck/MSD = Merck US = US-based Merck
- BMS = Bristol Myers Squibb = Bristol-Myers Squibb
- Roche = Genentech = Roche/Genentech

**Drug Name Expansions:**
- Nivolumab = nivo = opdivo = BMS-936558 = ONO-4538
- Ipilimumab = ipi = yervoy = BMS-734016 = MDX-010
- Pembrolizumab = pembro = keytruda = MK-3475
- Atezolizumab = atezo = tecentriq = MPDL3280A
- Vemurafenib = vemu = zelboraf = PLX4032
- Dabrafenib = dabra = tafinlar = GSK2118436
- Trametinib = trame = mekinist = GSK1120212

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
4. **Include flexible text matching** with multiple LIKE conditions for variations
5. **Order results meaningfully** (by efficacy, phase, company, etc.)
6. **Limit results appropriately** (10-20 unless specified otherwise)
7. **Double-check queries** before execution
8. **Present results in clear tabular format** with default parameters

SEARCH STRATEGY FOR ANY QUERY:
- Identify key terms and map to database columns
- Include all possible variations of terminology
- Search both treatment arms (Product/Regimen Name) and comparator arms (Comparator)
- Join relevant tables for comprehensive results
- Apply appropriate filters and sorting
- Present results with clinical context and interpretation

-  ** For Handle Comparator Queries only:**
    *   If you **do not** find the requested regimen in any Product/Regimen Name field, then check if it is mentioned in the Comparator field of any object.
    *   If you find it in a Comparator field, you MUST respond by:
        a. Stating clearly that the dataset **does not contain the specific performance data for the requested arm** (the comparator).
        b. Then, as a helpful alternative, state that you DO have the data for the treatment it was compared against.
        c. Provide the name of the Product/Regimen Name from that row and its corresponding value for the metric the user asked for.
        
RESPONSE FORMAT:
1. Understand the question and identify relevant clinical concepts
2. Query systematically using comprehensive terminology matching
3. Present results in clear tables with default parameters
4. Provide clinical interpretation and context
5. Specify data limitations or assumptions when relevant

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
