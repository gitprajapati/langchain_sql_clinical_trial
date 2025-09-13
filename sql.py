"""
Clinical Trial SQL Chatbot with Intelligent Visualization using LangGraph
Enhanced version with visualization capabilities using LangGraph nodes and Code 2 visual design.
"""

import os
import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, TypedDict
from dotenv import load_dotenv
import json
import re

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="Clinical Trial AI Assistant",
    page_icon="🔬",
    layout="wide",  # Changed to wide for better visualization
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean UI - matching code 2 style
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
    .css-1d391kg {
        display: none;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stChatInput > div {
        border-radius: 25px;
    }
    .visualization-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f0f2f6;
        margin-right: 20%;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# State definition for LangGraph
class GraphState(TypedDict):
    user_question: str
    sql_response: str
    visualization_intent: str
    dataframe_json: Optional[str]
    visualization_config: Optional[Dict[str, Any]]
    final_response: str
    error: Optional[str]

class VisualizationEngine:
    """Handles creation of clinical trial visualizations using Code 2 design pattern."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        self.llm_small = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    def create_dataframe_from_response(self, user_question: str, sql_response: str) -> Optional[str]:
        """Create DataFrame JSON from SQL response for visualization with proper NR handling."""
        df_prompt = PromptTemplate.from_template("""
        Extract the key data from this clinical trial response and create a structured DataFrame.
        
        USER QUESTION: {user_question}
        
        SQL RESPONSE: {sql_response}
        
        Create a JSON representation of a pandas DataFrame with the following requirements:
        1. Extract treatment names (regimens) and their corresponding metrics
        2. Include only numerical metrics that can be visualized (ORR, CR, PR, DCR, PFS, OS, etc.)
        3. Clean percentage values (remove % sign, convert to float)
        4. For survival endpoints (PFS, OS, DoR), preserve "NR" or "Not reached" values EXACTLY as "NR" - do NOT convert to null
        5. Handle other missing values as null
        6. Structure should be: {{"Treatment": [...], "Metric1": [...], "Metric2": [...]}}
        
        CRITICAL: For survival metrics (mPFS, mOS, mDoR), if you see "NR", "Not reached", "NE", or "Not estimable", 
        preserve these EXACTLY as they appear in the output JSON. Do not convert them to null or empty strings.
        
        Example output format:
        {{
            "Treatment/Trial Name": ["Treatment A", "Treatment B", "Treatment C"],
            "ORR": [45.0, 58.3, 47.2],
            "CR": [18.7, 22.0, 22.2],
            "PR": [26.3, 36.3, 25.0],
            "mPFS": [11.5, "NR", "NE"],
            "mOS": ["NR", "NR", 15.8]
        }}
        
        Only return the JSON object, no explanations.
        """)
        
        try:
            response = self.llm.invoke(df_prompt.format(
                user_question=user_question,
                sql_response=sql_response
            ))
            
            # Extract JSON from the response
            content = response.content.strip()
            
            # Try to find JSON in the response (in case the model adds extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Debug: Print the JSON before returning
                print(f"Generated JSON: {json_str}")
                
                # Validate the JSON can be parsed
                try:
                    test_data = json.loads(json_str)
                    print(f"JSON validation successful. Keys: {list(test_data.keys())}")
                    
                    # Check for NR values in the data
                    for key, values in test_data.items():
                        nr_count = sum(1 for v in values if str(v).upper() == 'NR')
                        if nr_count > 0:
                            print(f"Found {nr_count} NR values in column '{key}': {values}")
                    
                    return json_str
                except json.JSONDecodeError as e:
                    st.error(f"Generated invalid JSON: {e}")
                    return None
            else:
                st.error(f"Could not extract JSON from response: {content}")
                return None
                
        except Exception as e:
            st.error(f"Error creating dataframe: {e}")
            return None
    
    def detect_visualization_intent(self, user_question: str, sql_response: str) -> str:
        """Detect if user wants visualization based on question and response."""
        
        # Simple rule-based detection first
        comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'between', 'across', 'different']
        multiple_data_indicators = ['ORR', 'CR', 'PR', 'DCR', '%']
        
        question_lower = user_question.lower()
        response_lower = sql_response.lower()
        
        # Check if question asks for comparison
        has_comparison = any(keyword in question_lower for keyword in comparison_keywords)
        
        # Check if response has multiple numeric values
        percentage_count = response_lower.count('%')
        numeric_pattern_count = len(re.findall(r'\d+\.?\d*%?', sql_response))
        
        # Simple heuristic
        if has_comparison and percentage_count > 1:
            return "yes"
        elif numeric_pattern_count > 3:  # Multiple numeric values suggest tabular data
            return "yes"
        
        # Fallback to LLM if simple rules are inconclusive
        intent_prompt = PromptTemplate.from_template("""
        Analyze the user question and SQL response to determine if visualization would be helpful.
        
        USER QUESTION: {user_question}
        
        SQL RESPONSE: {sql_response}
        
        Return "yes" if:
        - User asks for comparison between treatments/trials
        - Response contains multiple numerical values that can be compared
        - Question involves efficacy metrics (ORR, PFS, OS, etc.) for multiple treatments
        - Question asks about trends, rankings, or comparisons
        - Response has tabular data that would benefit from charts
        
        Return "no" if:
        - Question is about single value lookup
        - Response is purely textual explanation
        - No comparative numerical data present
        - User asks for definitions or explanations
        
        Respond with ONLY "yes" or "no", nothing else.
        """)
        
        try:
            response = self.llm_small.invoke(intent_prompt.format(
                user_question=user_question,
                sql_response=sql_response
            ))
            result = response.content.strip().lower()
            return "yes" if "yes" in result else "no"
        except Exception as e:
            st.error(f"Error detecting visualization intent: {e}")
            return "no"
    
    def check_visualization_necessity(self, query: str, response: str) -> bool:
        """Enhanced logic from Code 2 to determine if visualization is needed."""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Strong indicators for visualization
        comparison_indicators = [
            'compare', 'comparison', 'versus', 'vs', 'vs.', 'against',
            'difference', 'better', 'best', 'worse', 'worst', 'superior', 'inferior',
            'efficacy', 'effectiveness', 'performance', 'which', 'what is the'
        ]
        
        # Check if query explicitly asks for comparison
        has_comparison_intent = any(indicator in query_lower for indicator in comparison_indicators)
        
        # Detect multiple treatments/trials in response
        trial_indicators = ['trial', 'study', 'checkmate', 'keynote', 'combi', 'columbus', 'cobrim', 
                        'dreamseq', 'inspire', 'relativity', 'phase', 'arm']
        
        multiple_treatments = sum(1 for indicator in trial_indicators if response_lower.count(indicator) > 1) >= 2
        
        # Check for tabular/structured data in response
        has_structured_data = (
            '|' in response and response.count('|') > 10 or  # Table format
            response_lower.count('orr') > 1 or
            response_lower.count('pfs') > 1 or
            response_lower.count('%') >= 3
        )
        
        # Check for numerical metrics
        import re
        numerical_patterns = [
            r'\d+\.?\d*%',           # Percentages
            r'\d+\.?\d*\s*months?',  # Time periods
            r'\d+\.?\d*\s*mg',       # Dosages
            r'hr[:\s]*\d+\.?\d*',    # Hazard ratios
            r'p[:\s]*[<>=]\s*\d+'    # P-values
        ]
        
        numerical_metrics = 0
        for pattern in numerical_patterns:
            numerical_metrics += len(re.findall(pattern, response_lower))
        
        has_sufficient_metrics = numerical_metrics >= 3
        
        # Enhanced decision logic
        should_visualize = (
            (has_comparison_intent and (multiple_treatments or has_structured_data)) or
            (has_structured_data and has_sufficient_metrics) or
            (multiple_treatments and has_sufficient_metrics)
        )
        
        return should_visualize
    
    def get_trial_arms_for_visualization(self, df_json: str) -> pd.DataFrame:
        """Convert JSON dataframe for visualization - Code 2 style."""
        try:
            if not df_json or df_json.strip() == "":
                return pd.DataFrame()
                
            data = json.loads(df_json)
            df = pd.DataFrame(data)
            
            if df.empty:
                return pd.DataFrame()
            
            # Prepare data similar to Code 2 style
            treatment_col = df.columns[0] if len(df.columns) > 0 else 'Treatment'
            
            # Create display names for treatments
            if treatment_col in df.columns:
                df['Display_Name'] = df[treatment_col].apply(lambda x: str(x)[:70])
            else:
                df['Display_Name'] = df.iloc[:, 0].apply(lambda x: str(x)[:70])
            
            return df
            
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error processing visualization data: {e}")
            return pd.DataFrame()
    
    def identify_relevant_metrics_from_context(self, query: str, response: str, df: pd.DataFrame) -> List[str]:
        """Enhanced metric identification to include all available clinical metrics."""
        potential_metrics = []
        
        # Define all possible clinical trial metrics based on your database schema
        clinical_metric_patterns = [
            # Efficacy metrics
            'ORR', 'CR', 'PR', 'DCR', 'mPFS', 'mOS', 'mDoR',
            # Survival rates
            '1-yr PFS Rate', '2-yr PFS Rate', '3-yr PFS Rate', '4-yr PFS Rate', '5-yr PFS Rate',
            '1-yr OS Rate', '2-yr OS Rate', '3-yr OS Rate', '4-yr OS Rate', '5-yr OS Rate',
            # Safety metrics
            'Gr 3/4 TRAEs', 'Gr ≥3 TRAEs', 'Gr 3/4 TEAEs', 'Gr ≥3 TEAEs',
            'Gr 3/4 AEs', 'Gr ≥3 AEs', 'Gr 3/4 irAEs', 'Gr ≥3 irAEs',
            # Additional metrics
            'Key AEs', 'Tx-related Deaths (Gr 5 TRAEs)', 'All Deaths (Gr 5 AEs)',
            'PFS HR (p-value)', 'OS HR (p-value)', 'N', 'n', 'Median Follow-up (mFU)'
        ]
        
        # Check each column in the dataframe
        for col in df.columns:
            col_lower = col.lower()
            
            # Skip obvious non-metric columns
            skip_patterns = ['id', 'name', 'url', 'note', 'comment', 'source', 'tag', 
                           'display_name', 'treatment', 'trial', 'acronym', 'developer', 
                           'phase', 'biomarker', 'histology', 'indication']
            
            if any(skip in col_lower for skip in skip_patterns):
                continue
            
            # Include if it matches clinical metric patterns exactly
            if col in clinical_metric_patterns:
                potential_metrics.append(col)
                continue
            
            # Include if it contains clinical metric keywords
            metric_keywords = [
                'orr', 'cr', 'pr', 'dcr', 'pfs', 'os', 'dor', 'rate', 'hr', 'survival',
                'response', 'trae', 'teae', 'aes', 'death', 'follow-up', 'mfu',
                'grade', 'gr', 'adverse', 'toxic', 'safety'
            ]
            
            if any(keyword in col_lower for keyword in metric_keywords):
                # Check if column has meaningful data (not all empty/NA)
                non_empty_count = df[col].dropna().astype(str).str.strip()
                non_empty_count = non_empty_count[non_empty_count != ''].shape[0]
                
                if non_empty_count > 0:
                    potential_metrics.append(col)
        
        if not potential_metrics:
            return []
        
        # Parse response to find mentioned metrics
        response_metrics = []
        response_lower = response.lower()
        
        # Look for metrics mentioned in the response text
        for metric in potential_metrics:
            metric_variations = [metric.lower(), metric.replace(' ', '').lower(), 
                               metric.replace('/', '').lower(), metric.replace('≥', '').lower()]
            
            # Special cases for common abbreviations
            if 'ORR' in metric:
                metric_variations.extend(['overall response rate', 'response rate'])
            elif 'mPFS' in metric:
                metric_variations.extend(['median progression-free survival', 'progression-free survival'])
            elif 'mOS' in metric:
                metric_variations.extend(['median overall survival', 'overall survival'])
            elif 'TRAEs' in metric:
                metric_variations.extend(['treatment-related adverse events', 'treatment related adverse events'])
            
            if any(var in response_lower for var in metric_variations):
                response_metrics.append(metric)
        
        # Prioritization logic
        # 1. If metrics are mentioned in response, prioritize those
        if response_metrics:
            final_metrics = response_metrics[:8]  # Show more metrics if mentioned
        else:
            # 2. Default prioritization based on clinical importance
            priority_order = [
                'ORR', 'mPFS', 'mOS', 'CR', 'PR', 'DCR',  # Primary efficacy
                'Gr 3/4 TRAEs', 'Gr ≥3 TRAEs',            # Safety
                '1-yr PFS Rate', '2-yr PFS Rate',          # Survival rates
                '1-yr OS Rate', '2-yr OS Rate',
                'mDoR', 'Key AEs'                          # Secondary
            ]
            
            final_metrics = []
            for priority_metric in priority_order:
                if priority_metric in potential_metrics:
                    final_metrics.append(priority_metric)
                if len(final_metrics) >= 6:  # Limit to 6 for clean display
                    break
            
            # Add any remaining metrics if we don't have enough
            if len(final_metrics) < 4:
                remaining = [m for m in potential_metrics if m not in final_metrics][:4-len(final_metrics)]
                final_metrics.extend(remaining)
        
        print(f"Available columns in df: {list(df.columns)}")
        print(f"Identified potential metrics: {potential_metrics}")
        print(f"Final selected metrics: {final_metrics}")
        
        return final_metrics
    
    def display_bar_charts(self, df: pd.DataFrame, trial_col: str, metric_cols: List[str], key_prefix: str = ""):
        """Display bar charts design pattern with proper NR and NE handling."""
        if df.empty:
            st.info("No data available for visualization")
            return
        
        # Create display names - Code 2 style
        if 'Display_Name' not in df.columns:
            df['Display_Name'] = df[trial_col].apply(lambda x: str(x)[:70])
        
        df_viz = df.copy()
        
        # Prepare data for melting - only include available metrics
        available_metrics = [col for col in metric_cols if col in df_viz.columns]
        if not available_metrics:
            st.info("No metrics available for visualization")
            return
        
        # Melt data for plotting - Code 2 style with debugging
        melted = pd.melt(df_viz[['Display_Name'] + available_metrics], 
                        id_vars='Display_Name',
                        var_name="Metric", 
                        value_name="RawValue")
        
        # Debug: Print the raw values before processing
        print("Raw values in melted data:")
        print(melted[['Display_Name', 'Metric', 'RawValue']].head(10))
        
        # Clean and process values - Code 2 style
        melted["RawValue"] = melted["RawValue"].astype(str).str.strip()
        
        # Define missing values - EXCLUDE "NR", "NE", and "Not Reached" from missing values
        missing_values = {"", "na", "n/a", "nan", "none", "null", "not available"}
        # Define "Not Reached" and "Not Estimable" values separately
        not_reached_values = {"nr", "not reached", "not_reached", "notreached", "not-reached"}
        not_estimable_values = {"ne", "not estimable", "not_estimable", "notestimable", "not-estimable"}
        
        melted["RawValue_lower"] = melted["RawValue"].str.lower().str.strip()
        melted["IsMissing"] = melted["RawValue_lower"].isin(missing_values)
        melted["IsNotReached"] = melted["RawValue_lower"].isin(not_reached_values)
        melted["IsNotEstimable"] = melted["RawValue_lower"].isin(not_estimable_values)
        
        # Extract numeric values - Code 2 style
        melted["Value"] = melted["RawValue"].str.replace('%', '', regex=False)
        melted["Value"] = melted["Value"].str.replace('months', '', regex=False)
        melted["Value"] = melted["Value"].str.extract(r'([\d.]+)', expand=False)
        melted["Value"] = pd.to_numeric(melted["Value"], errors='coerce')
        
        # Set plot values - Code 2 approach
        melted["PlotValue"] = melted["Value"].fillna(0)
        melted.loc[melted["IsMissing"], "PlotValue"] = 0.1
        melted.loc[melted["IsNotReached"], "PlotValue"] = 0.1
        melted.loc[melted["IsNotEstimable"], "PlotValue"] = 0.1
        
        # Create display text with proper NR and NE handling - Enhanced Code 2 style
        def create_display_text(row):
            raw_val = str(row["RawValue"]).strip()
            raw_val_lower = raw_val.lower().strip()
            
            # Debug print to see what values we're getting
            print(f"Processing value: '{raw_val}' -> '{raw_val_lower}'")
            
            # First check for explicit NR patterns (Code 2 approach) - MOST COMPREHENSIVE
            if (raw_val_lower == 'nr' or 
                raw_val.upper() == 'NR' or
                raw_val_lower == 'not reached' or
                raw_val_lower == 'not_reached' or
                raw_val_lower == 'notreached' or
                raw_val_lower == 'not-reached' or
                'not reached' in raw_val_lower or
                raw_val_lower.startswith('nr') or
                raw_val_lower.endswith('nr')):
                print(f"Detected NR value: {raw_val}")
                return "NR"
            
            # Check for explicit NE patterns (Not Estimable)
            elif (raw_val_lower == 'ne' or 
                raw_val.upper() == 'NE' or
                raw_val_lower == 'not estimable' or
                raw_val_lower == 'not_estimable' or
                raw_val_lower == 'notestimable' or
                raw_val_lower == 'not-estimable' or
                'not estimable' in raw_val_lower):
                print(f"Detected NE value: {raw_val}")
                return "NE"
            
            # Check for missing/empty values
            elif (raw_val_lower in ['', 'na', 'n/a', 'nan', 'none', 'null', 'not available'] or
                raw_val_lower == 'nan' or 
                pd.isna(raw_val) or
                raw_val == '' or
                raw_val == 'none' or
                str(raw_val) == 'nan'):
                print(f"Detected missing value: {raw_val}")
                return "N/A"
            
            # Regular numeric or text values
            else:
                cleaned_val = raw_val.upper().replace('MONTHS', '').replace('MONTH', '').strip()
                print(f"Regular value: {raw_val} -> {cleaned_val}")
                return cleaned_val
        
        melted["DisplayText"] = melted.apply(create_display_text, axis=1)
        
        # Additional safety check for survival metrics - Code 2 approach enhanced for NE
        survival_metrics = melted["Metric"].str.contains("OS|PFS|DoR|DOR", case=False, na=False)
        potentially_special = (melted["DisplayText"] == "N/A") & survival_metrics
        
        # If we find survival metrics showing N/A, check if raw value could be NR or NE
        for idx in melted[potentially_special].index:
            raw_val = str(melted.loc[idx, "RawValue"]).strip().upper()
            if raw_val in ['NR', 'NOT REACHED', 'NOTREACHED', 'NOT_REACHED']:
                melted.loc[idx, "DisplayText"] = "NR"
            elif raw_val in ['NE', 'NOT ESTIMABLE', 'NOTESTIMABLE', 'NOT_ESTIMABLE']:
                melted.loc[idx, "DisplayText"] = "NE"
        
        # Create color scheme - Code 2 style
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        arm_names = melted['Display_Name'].unique()
        
        while len(colors) < len(arm_names):
            colors.extend(colors)
        
        arm_color_map = {name: colors[i % len(colors)] for i, name in enumerate(arm_names)}
        
        # Calculate dimensions - Code 2 style
        num_metrics = len(available_metrics)
        num_arms = len(df_viz)
        
        if num_arms <= 2:
            base_height_per_arm = 60
            min_height = 300
        elif num_arms <= 4:
            base_height_per_arm = 80
            min_height = 400
        else:
            base_height_per_arm = 100
            min_height = 500
        
        chart_height = max(min_height, base_height_per_arm * num_arms + 150)
        
        # Create bar chart - Code 2 style
        fig = px.bar(
            melted,
            x="PlotValue",
            y="Display_Name",
            color="Display_Name",
            facet_col="Metric",
            facet_col_wrap=min(num_metrics, 4),
            orientation="h",
            text="DisplayText",
            color_discrete_map=arm_color_map,
            title=f"Clinical Metrics Comparison"
        )
        
        # Update traces - Code 2 style
        fig.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="black"),
            cliponaxis=False,
            marker=dict(
                line=dict(width=1, color='rgba(0,0,0,0.2)')
            )
        )
        
        # Layout updates - Code 2 style
        fig.update_layout(
            height=chart_height,
            showlegend=False,
            margin=dict(l=400, r=150, t=100, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white",
            bargap=0.3 if num_arms <= 2 else 0.2,
            font=dict(color="black")
        )
        
        # Clean facet titles - Code 2 style
        fig.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1], 
            font=dict(size=12, color="black")
        ))
        
        fig.for_each_xaxis(lambda x: x.update(
            title='', 
            showticklabels=False,
            gridcolor="rgba(0,0,0,0.1)"
        ))
        
        fig.for_each_yaxis(lambda y: y.update(
            title='',
            tickfont=dict(size=9, color="black"),
            tickmode='linear',
            automargin=True,
            tickangle=0
        ))
        
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

class ClinicalTrialChatbot:
    """Enhanced Clinical trial chatbot with visualization capabilities."""
    
    def __init__(self):
        self.db_path = "clinical_trials.db"
        self.db = None
        self.sql_agent = None
        self.viz_engine = VisualizationEngine()
        self.memory = MemorySaver()
        self.workflow = None
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
        """Initialize database connection, SQL agent, and workflow."""
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
            llm = ChatOpenAI(model="gpt-4.1", temperature=0)
            
            # Create toolkit with all SQL tools
            toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
            tools = toolkit.get_tools()
            
            # Create system prompt for clinical trial domain
            system_prompt = self.get_clinical_trial_system_prompt()
            
            # Create SQL agent
            from langgraph.prebuilt import create_react_agent
            self.sql_agent = create_react_agent(
                llm, 
                tools, 
                prompt=system_prompt,
                checkpointer=self.memory
            )
            
            # Initialize workflow
            self.workflow = self._create_workflow()
            
            st.success("✅ Clinical Trial SQL Agent with Visualization initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing components: {e}")
            st.stop()
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for intelligent visualization."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("sql_query", self._sql_query_node)
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("create_dataframe", self._create_dataframe_node)
        workflow.add_node("finalize_response", self._finalize_response_node)
        
        # Add edges
        workflow.set_entry_point("sql_query")
        workflow.add_edge("sql_query", "intent_detection")
        
        # Conditional edge based on visualization intent
        workflow.add_conditional_edges(
            "intent_detection",
            self._should_visualize,
            {
                "yes": "create_dataframe",
                "no": "finalize_response"
            }
        )
        
        workflow.add_edge("create_dataframe", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _sql_query_node(self, state: GraphState) -> GraphState:
        """Node to execute SQL query using the existing SQL agent."""
        try:
            config = {"configurable": {"thread_id": "sql_thread"}}
            
            messages = []
            for step in self.sql_agent.stream(
                {"messages": [{"role": "user", "content": state["user_question"]}]},
                config,
                stream_mode="values"
            ):
                messages = step["messages"]
            
            if messages and len(messages) > 0:
                final_message = messages[-1]
                sql_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
                state["sql_response"] = sql_response
            else:
                state["sql_response"] = "No response generated."
                
        except Exception as e:
            state["error"] = f"Error in SQL query: {str(e)}"
            state["sql_response"] = f"Error processing query: {str(e)}"
        
        return state
    
    def _intent_detection_node(self, state: GraphState) -> GraphState:
        """Node to detect visualization intent."""
        try:
            intent = self.viz_engine.check_visualization_necessity(
                state["user_question"], 
                state["sql_response"]
            )
            state["visualization_intent"] = "yes" if intent else "no"
        except Exception as e:
            state["error"] = f"Error detecting intent: {str(e)}"
            state["visualization_intent"] = "no"
        
        return state
    
    def _create_dataframe_node(self, state: GraphState) -> GraphState:
        """Node to create DataFrame for visualization."""
        try:
            df_json = self.viz_engine.create_dataframe_from_response(
                state["user_question"],
                state["sql_response"]
            )
            state["dataframe_json"] = df_json
        except Exception as e:
            state["error"] = f"Error creating dataframe: {str(e)}"
            state["dataframe_json"] = None
        
        return state
    
    def _finalize_response_node(self, state: GraphState) -> GraphState:
        """Node to finalize the response."""
        state["final_response"] = state["sql_response"]
        return state
    
    def _should_visualize(self, state: GraphState) -> str:
        """Conditional function to determine if visualization should be created."""
        return state.get("visualization_intent", "no")
    
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
6. **trial_summary** VIEW - Pre-joined key metrics (regimen_name, phase, orr, pfs, os, safety_grade3_4)
7. **efficacy_summary** VIEW - Clean efficacy metrics with standardized column names
8. **safety_summary** VIEW - Clean safety data with standardized column names
9. Distinguish between active and completed phases this would be the "Highest Phase" column:
    9.1 - Active Development: This includes Ph0, Ph1, Ph2, Ph3, and Ph4, along with their various combinations (e.g., Ph1/2, Ph2/3, Ph3b/4).
    9.2 - Completed Development: This primarily refers to the "Approved" status, often with regional specifics like "(US)" or "(EU)".

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

FILTERING RULES FOR ANTI-PD-1 QUERIES:
- When asked for unique Anti-PD-1 agents:
    * Only include products/regimens where "Regimen MoAs" OR "Product/Regimen Target" contains 'Anti-PD-1'.
    * Exclude agents where the MoA/Target does not explicitly include 'Anti-PD-1' (e.g., Anti-CTLA-4, Anti-TIGIT, Anti-PD-L1, Anti-LAG-3, etc...).
    * If a regimen is a combination, only extract and report the Anti-PD-1 antibody from it.
    * Deduplicate results so each Anti-PD-1 antibody is listed only once.
- Example:
    * "Nurulimab + Prolgolimab" → Only Prolgolimab should appear (Anti-PD-1).
    * "Porustobart + Toripalimab" → Only Toripalimab should appear (Anti-PD-1).

RESPONSE FORMAT:
1. Understand the question and identify relevant clinical concepts
2. Query systematically using comprehensive terminology matching
3. Provide clinical interpretation and context
4. Specify data limitations or assumptions when relevant
5. Create table where ever needed for comparisions.
6. Give all trial info in reponse if asked for comparision until specified.

Remember: Handle ANY type of clinical trial question flexibly by mapping user terminology to database fields and using comprehensive search strategies across all relevant columns and tables.
"""
    
    def process_query(self, question: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """Process query through the workflow and return response with optional visualization data."""
        try:
            # Initial state
            initial_state = GraphState(
                user_question=question,
                sql_response="",
                visualization_intent="no",
                dataframe_json=None,
                visualization_config=None,
                final_response="",
                error=None
            )
            
            # Run workflow
            config = {"configurable": {"thread_id": "viz_workflow"}}
            final_state = self.workflow.invoke(initial_state, config)
            
            # Prepare visualization data if needed
            viz_data = None
            if (final_state["visualization_intent"] == "yes" and 
                final_state["dataframe_json"] is not None):
                
                # Get dataframe from JSON
                viz_df = self.viz_engine.get_trial_arms_for_visualization(final_state["dataframe_json"])
                
                if not viz_df.empty:
                    # Get relevant metrics
                    viz_metrics = self.viz_engine.identify_relevant_metrics_from_context(
                        question, final_state["final_response"], viz_df
                    )
                    
                    if viz_metrics:
                        viz_data = {
                            'df': viz_df,
                            'trial_col': 'Display_Name',
                            'metrics': viz_metrics,
                            'query': question
                        }
            
            return final_state["final_response"], viz_data
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            return error_msg, None

def create_visualization(viz_data: Dict[str, Any], message_idx: int):
    """Create interactive visualization using Code 2 design pattern."""
    if not viz_data:
        return
    
    df = viz_data['df']
    default_metrics = viz_data['metrics']
    
    if df.empty:
        return
    
    # Get available metrics - similar to Code 2
    all_possible_metrics = [
        "ORR", "CR", "PR", "mPFS", "mOS", "DCR", "mDoR",
        "1-yr PFS Rate", "2-yr PFS Rate", "3-yr PFS Rate", "4-yr PFS Rate", "5-yr PFS Rate",
        "1-yr OS Rate", "2-yr OS Rate", "3-yr OS Rate", "4-yr OS Rate", "5-yr OS Rate",
        "Gr 3/4 TRAEs", "Gr ≥3 TRAEs", "Gr 3/4 TEAEs", "Gr ≥3 TEAEs"
    ]
    
    available_metrics = [m for m in all_possible_metrics if m in df.columns and 
                        df[m].dropna().astype(str).str.strip().str.lower().ne('na').any()]
    
    if available_metrics:
        valid_default_metrics = [m for m in default_metrics if m in available_metrics] or available_metrics[:4]
        
        selected_metrics = st.multiselect(
            "📈 **Select metrics to compare:**",
            options=available_metrics, 
            default=valid_default_metrics,
            key=f"metrics_selector_{message_idx}"
        )
        
        if selected_metrics:
            # Use the visualization engine from this class
            viz_engine = st.session_state.chatbot.viz_engine
            viz_engine.display_bar_charts(df, 'Display_Name', selected_metrics, f"viz_{message_idx}")
        else:
            st.warning("⚠️ Please select at least one metric to visualize.")
    else:
        st.warning("⚠️ No metrics with valid data are available for visualization.")

def main():
    """Main application function."""
    
    # Clean header section - Code 2 style
    st.markdown('<h1 class="main-header">🔬 Clinical Trial AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about clinical trial data with intelligent visualizations</p>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing Clinical Trial SQL Agent with Visualization..."):
            st.session_state.chatbot = ClinicalTrialChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualization if present
            if message["role"] == "assistant" and "viz_data" in message:
                if message["viz_data"] is not None:
                    with st.container():
                        st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                        create_visualization(message["viz_data"], i)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about clinical trials..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical trial data..."):
                response, viz_data = st.session_state.chatbot.process_query(prompt)
                st.markdown(response)
                
                # Display visualization if created
                if viz_data is not None:
                    with st.container():
                        st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                        create_visualization(viz_data, len(st.session_state.messages))
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store message with visualization data
        message_data = {
            "role": "assistant", 
            "content": response,
            "viz_data": viz_data
        }
        st.session_state.messages.append(message_data)

if __name__ == "__main__":
    main()
