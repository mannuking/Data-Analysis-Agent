# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, io, re, json, base64
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import altair as alt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Union, Optional
from dotenv import load_dotenv
import pydeck as pdk
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.chart_container import chart_container
from streamlit_extras.colored_header import colored_header
import openpyxl
import xlrd

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    layout="wide", 
    page_title="Data Analysis Agent", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Set Plotly as default visualization tool
pio.templates.default = "plotly_white"

# Load environment variables
load_dotenv()

# === Configuration ===
api_key = os.environ.get("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# CSS for enhanced UI
def load_custom_css():
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        .css-1d391kg {padding-top: 1rem;}
        .stTabs [data-baseweb="tab-list"] {justify-content: center;}
        .stTabs [data-baseweb="tab"] {font-size: 1.1rem;}
        .sidebar .sidebar-content {background: #f5f7fa;}
        .css-1544g2n {padding-top: 0rem;}
        .st-emotion-cache-1wrcr25 {margin-bottom: 0.5rem;}
        .st-emotion-cache-6qob1r {gap: 0.5rem;}
        .thinking {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        .viz-container {
            background-color: white;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
        }
        .main-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #262730;
        }
        .sub-header {
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: #404254;
        }
        .chart-action-btn {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 0.5rem;
            margin-right: 0.5rem;
            font-size: 0.8rem;
            border: 1px solid #e0e0e0;
        }
        /* Adjust sidebar width */
        [data-testid="stSidebar"] {
            min-width: 450px !important;
            max-width: 450px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> Dict[str, Any]:
    """Analyze the query to determine what type of visualization or insight is being requested."""
    # Use LLM to understand intent with more detail
    messages = [
        {"role": "system", "content": """detailed thinking off. You are an assistant that analyzes data visualization queries. 
        Return a structured JSON object with the following fields:
        - visualization_type: The type of visualization that would best answer this query (line, bar, pie, scatter, heatmap, table, map, none)
        - analysis_type: The type of analysis being requested (trend, comparison, distribution, correlation, relationship, composition, geographic, summary, drill_down)
        - fields_needed: Array of likely data fields needed based on the query
        - time_component: Boolean indicating if there is a time dimension to the analysis
        - aggregation: The type of aggregation if applicable (sum, average, count, min, max, etc.)
        - filters: Any filtering conditions detected in the query
        - complexity: Rate the complexity of the query from 1-5
        """},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        # Fallback to basic intent detection if JSON parsing fails
        return {
            "visualization_type": "table" if "table" in query.lower() else "bar",
            "analysis_type": "summary",
            "fields_needed": [],
            "time_component": False,
            "aggregation": "count",
            "filters": {},
            "complexity": 1
        }

# ------------------  DataLoaderTool ---------------------------
def DataLoaderTool(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from various file formats and provide basic analysis of the dataset."""
    try:
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        # Generate dataset metadata
        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist(),
            "text_columns": [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().max() > 50]
        }
        
        # Add date detection for columns that might be dates but aren't recognized
        for col in df.columns:
            if col not in metadata["datetime_columns"] and df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    # If no error, try converting the column and add to datetime columns
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():
                        metadata["datetime_columns"].append(col)
                except:
                    pass
        
        # Detect potential ID columns
        metadata["id_columns"] = [col for col in df.columns if ('id' in col.lower() or 'key' in col.lower()) and df[col].nunique() > df.shape[0] * 0.8]
        
        return df, metadata
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}

# ------------------  DataProcessorTool ---------------------------
def DataProcessorTool(df: pd.DataFrame, metadata: Dict[str, Any], query_analysis: Dict[str, Any]) -> pd.DataFrame:
    """Process dataframe based on query analysis to prepare for visualization."""
    processed_df = df.copy()
    
    # Handle datetime conversions for time-based queries
    if query_analysis.get("time_component", False):
        for col in metadata["datetime_columns"]:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
    
    # Apply filters if specified in query
    filters = query_analysis.get("filters", {})
    for col, filter_value in filters.items():
        if col in processed_df.columns:
            if isinstance(filter_value, list):
                processed_df = processed_df[processed_df[col].isin(filter_value)]
            elif isinstance(filter_value, dict) and "range" in filter_value:
                low, high = filter_value["range"]
                processed_df = processed_df[(processed_df[col] >= low) & (processed_df[col] <= high)]
            else:
                processed_df = processed_df[processed_df[col] == filter_value]
    
    # Handle missing values based on query complexity
    complexity = query_analysis.get("complexity", 3)
    if complexity <= 2:
        # For simple queries, just drop rows with NaN in relevant columns
        fields_needed = query_analysis.get("fields_needed", [])
        if fields_needed:
            relevant_cols = [col for col in fields_needed if col in processed_df.columns]
            if relevant_cols:
                processed_df = processed_df.dropna(subset=relevant_cols)
    else:
        # For complex queries, use more sophisticated handling
        numeric_cols = metadata.get("numeric_columns", [])
        for col in numeric_cols:
            if processed_df[col].isna().any():
                # Fill numeric NaNs with median
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        for col in processed_df.columns:
            if col not in numeric_cols and processed_df[col].isna().any():
                # Fill categorical NaNs with mode or "Unknown"
                if processed_df[col].dropna().shape[0] > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                else:
                    processed_df[col] = processed_df[col].fillna("Unknown")
    
    return processed_df

# ------------------  VisualizationTool ---------------------------
def VisualizationTool(df: pd.DataFrame, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create appropriate visualizations based on query analysis and dataframe."""
    viz_type = query_analysis.get("visualization_type", "bar")
    analysis_type = query_analysis.get("analysis_type", "summary")
    fields = query_analysis.get("fields_needed", [])
    time_component = query_analysis.get("time_component", False)
    aggregation = query_analysis.get("aggregation", "count")
    
    # Identify the most appropriate fields if not specified
    if not fields or not all(field in df.columns for field in fields):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Auto-select appropriate fields based on visualization type
        if viz_type == "bar":
            x_candidates = categorical_cols[:1] if categorical_cols else df.columns[:1].tolist()
            y_candidates = numeric_cols[:1] if numeric_cols else []
            fields = x_candidates + y_candidates
        elif viz_type == "line":
            x_candidates = datetime_cols[:1] if datetime_cols else df.columns[:1].tolist()
            y_candidates = numeric_cols[:1] if numeric_cols else []
            fields = x_candidates + y_candidates
        elif viz_type == "scatter":
            fields = numeric_cols[:2] if len(numeric_cols) >= 2 else df.columns[:2].tolist()
        elif viz_type == "pie":
            if categorical_cols and numeric_cols:
                fields = [categorical_cols[0], numeric_cols[0]]
            else:
                fields = df.columns[:2].tolist()
        elif viz_type == "heatmap":
            if len(categorical_cols) >= 2:
                fields = categorical_cols[:2]
            else:
                fields = df.columns[:2].tolist()
        elif viz_type == "map":
            # Look for geographic columns
            geo_candidates = [col for col in df.columns if any(geo_term in col.lower() for geo_term in 
                              ["country", "state", "city", "region", "latitude", "longitude", "lat", "long", "zip", "postal"])]
            fields = geo_candidates[:2] if geo_candidates else df.columns[:2].tolist()
        elif viz_type == "table":
            fields = df.columns[:5].tolist()
    
    # Create visualization based on type
    result = {"type": viz_type, "figure": None, "fig_type": None, "data": None}
    
    try:
        if viz_type == "bar":
            # Determine if we need grouped or stacked bars
            if len(fields) >= 3 and fields[2] in df.columns:
                # Create grouped bar chart
                fig = px.bar(
                    df, x=fields[0], y=fields[1], color=fields[2],
                    title=f"Bar Chart of {fields[1]} by {fields[0]} and {fields[2]}",
                    barmode="group",
                    height=500
                )
            elif len(fields) >= 2 and fields[0] in df.columns and fields[1] in df.columns:
                # Basic bar chart with numeric y-axis
                if df[fields[1]].dtype.kind in 'bifc':  # Check if y is numeric
                    fig = px.bar(
                        df, x=fields[0], y=fields[1],
                        title=f"Bar Chart of {fields[1]} by {fields[0]}",
                        height=500
                    )
                else:  # Count-based bar chart
                    counts = df[fields[0]].value_counts().reset_index()
                    counts.columns = [fields[0], 'count']
                    fig = px.bar(
                        counts, x=fields[0], y='count',
                        title=f"Count of {fields[0]}",
                        height=500
                    )
            else:
                # Fallback to count-based single column
                field = fields[0] if fields and fields[0] in df.columns else df.columns[0]
                counts = df[field].value_counts().reset_index()
                counts.columns = [field, 'count']
                fig = px.bar(
                    counts, x=field, y='count',
                    title=f"Count of {field}",
                    height=500
                )
            
            # Apply theme and layout improvements
            fig.update_layout(
                plot_bgcolor='white',
                margin=dict(t=50, l=10, r=10, b=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            result["figure"] = fig
            result["fig_type"] = "plotly"
            
        elif viz_type == "line":
            if len(fields) >= 2 and fields[0] in df.columns and fields[1] in df.columns:
                # Convert to datetime if needed and possible
                if not pd.api.types.is_datetime64_any_dtype(df[fields[0]]):
                    try:
                        df[fields[0]] = pd.to_datetime(df[fields[0]], errors='coerce')
                    except:
                        pass
                
                # If we have a third field, use it for grouping
                if len(fields) >= 3 and fields[2] in df.columns:
                    fig = px.line(
                        df, x=fields[0], y=fields[1], color=fields[2],
                        title=f"Trend of {fields[1]} over {fields[0]} by {fields[2]}",
                        height=500
                    )
                else:
                    fig = px.line(
                        df, x=fields[0], y=fields[1],
                        title=f"Trend of {fields[1]} over {fields[0]}",
                        height=500
                    )
                
                # Apply theme and layout improvements
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(t=50, l=10, r=10, b=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                result["figure"] = fig
                result["fig_type"] = "plotly"
                
        elif viz_type == "pie":
            if len(fields) >= 2 and fields[0] in df.columns and fields[1] in df.columns:
                # Use specified value field if numeric
                if df[fields[1]].dtype.kind in 'bifc':
                    fig = px.pie(
                        df, names=fields[0], values=fields[1],
                        title=f"Distribution of {fields[1]} by {fields[0]}",
                        height=500
                    )
                else:
                    # Count-based pie chart
                    counts = df[fields[0]].value_counts().reset_index()
                    counts.columns = [fields[0], 'count']
                    fig = px.pie(
                        counts, names=fields[0], values='count',
                        title=f"Distribution of {fields[0]}",
                        height=500
                    )
            else:
                # Fallback to single column count
                field = fields[0] if fields and fields[0] in df.columns else df.columns[0]
                counts = df[field].value_counts().reset_index()
                counts.columns = [field, 'count']
                fig = px.pie(
                    counts, names=field, values='count',
                    title=f"Distribution of {field}",
                    height=500
                )
            
            # Enhance the pie chart
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                margin=dict(t=50, l=10, r=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            result["figure"] = fig
            result["fig_type"] = "plotly"
        
        elif viz_type == "scatter":
            if len(fields) >= 2 and fields[0] in df.columns and fields[1] in df.columns:
                # Basic scatter plot
                if len(fields) >= 3 and fields[2] in df.columns:
                    # With color dimension
                    fig = px.scatter(
                        df, x=fields[0], y=fields[1], color=fields[2],
                        title=f"Scatter Plot of {fields[1]} vs {fields[0]} by {fields[2]}",
                        height=500
                    )
                else:
                    fig = px.scatter(
                        df, x=fields[0], y=fields[1],
                        title=f"Scatter Plot of {fields[1]} vs {fields[0]}",
                        height=500
                    )
                
                # Apply theme and layout improvements
                fig.update_layout(
                    plot_bgcolor='white',
                    margin=dict(t=50, l=10, r=10, b=10),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                result["figure"] = fig
                result["fig_type"] = "plotly"
        
        elif viz_type == "heatmap":
            if len(fields) >= 2 and fields[0] in df.columns and fields[1] in df.columns:
                # Create pivot table for heatmap
                if len(fields) >= 3 and fields[2] in df.columns:
                    # Use third field for values
                    pivot_df = df.pivot_table(
                        index=fields[0], columns=fields[1], values=fields[2],
                        aggfunc=aggregation
                    )
                else:
                    # Count occurrences
                    pivot_df = df.pivot_table(
                        index=fields[0], columns=fields[1], aggfunc='count'
                    )
                    # Get the first column of the MultiIndex columns if needed
                    if isinstance(pivot_df.columns, pd.MultiIndex):
                        pivot_df = pivot_df[pivot_df.columns.levels[0][0]]
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df, 
                    title=f"Heatmap of {fields[0]} vs {fields[1]}",
                    height=600,
                    color_continuous_scale="Viridis"
                )
                
                fig.update_layout(
                    margin=dict(t=50, l=10, r=10, b=10)
                )
                result["figure"] = fig
                result["fig_type"] = "plotly"
        
        elif viz_type == "map":
            # Check for lat/long columns
            lat_cols = [col for col in df.columns if any(term in col.lower() for term in ["latitude", "lat"])]
            long_cols = [col for col in df.columns if any(term in col.lower() for term in ["longitude", "long", "lng"])]
            
            if lat_cols and long_cols:
                # Create map with lat/long coordinates
                lat_col, long_col = lat_cols[0], long_cols[0]
                # Try to find a numeric column for color
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                color_col = numeric_cols[0] if numeric_cols else None
                
                if color_col:
                    fig = px.scatter_mapbox(
                        df, lat=lat_col, lon=long_col, color=color_col,
                        title=f"Map of {color_col} by Location",
                        height=600,
                        zoom=3
                    )
                else:
                    fig = px.scatter_mapbox(
                        df, lat=lat_col, lon=long_col,
                        title="Location Map",
                        height=600,
                        zoom=3
                    )
                
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                result["figure"] = fig
                result["fig_type"] = "plotly"
            else:
                # Check for country/region columns
                geo_cols = [col for col in df.columns if any(term in col.lower() for term in 
                           ["country", "state", "city", "region", "county"])]
                
                if geo_cols:
                    # Create choropleth map
                    geo_col = geo_cols[0]
                    # Try to find a numeric column for color
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    color_col = numeric_cols[0] if numeric_cols else None
                    
                    if color_col:
                        # Create aggregated data by region
                        geo_df = df.groupby(geo_col)[color_col].mean().reset_index()
                        fig = px.choropleth(
                            geo_df, locations=geo_col, locationmode="country names",
                            color=color_col, title=f"Map of {color_col} by {geo_col}",
                            height=600,
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                        result["figure"] = fig
                        result["fig_type"] = "plotly"
                    else:
                        # Fallback to table
                        result["data"] = df[fields] if fields else df.head(10)
                        result["fig_type"] = "table"
                else:
                    # Fallback to table
                    result["data"] = df[fields] if fields else df.head(10)
                    result["fig_type"] = "table"
        
        elif viz_type == "table":
            # Return dataframe for table display
            result["data"] = df[fields] if fields else df.head(10)
            result["fig_type"] = "table"
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        # Fallback to table
        result["data"] = df.head(10)
        result["fig_type"] = "table"
    
    return result

# === CodeGeneration TOOLS ============================================

# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols: List[str], query: str, query_analysis: Dict[str, Any]) -> str:
    """Generate a prompt for the LLM to write advanced visualization code based on the query and columns."""
    viz_type = query_analysis.get("visualization_type", "bar")
    
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using Plotly to answer this query:
    "{query}"

    Based on query analysis, a {viz_type} chart would be most appropriate.

    Rules
    -----
    1. Use plotly.express or plotly.graph_objects for creating interactive visualizations.
    2. Assign the final plotly Figure object to a variable named `result`.
    3. Add appropriate title, labels, and styling to make the visualization professional.
    4. Handle potential errors or edge cases in the data.
    5. If the query can't be answered with a {viz_type} chart, use your judgment to create the most appropriate visualization.
    6. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """

# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols: List[str], query: str, query_analysis: Dict[str, Any]) -> str:
    """Generate a prompt for the LLM to write advanced data analysis code (no plotting)."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code to answer this query:
    "{query}"

    Rules
    -----
    1. Use pandas, numpy, and scikit-learn if needed for data analysis.
    2. Perform any necessary data cleaning, aggregation, filtering, or transformation.
    3. Assign the final result (DataFrame, Series, or scalar) to a variable named `result`.
    4. Ensure the `result` contains only the information needed to answer the query, not the entire dataset.
    5. Handle potential errors, missing values, or edge cases.
    6. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Analyzes the query, selects the appropriate code generation tool, and gets code from the LLM."""
    # First, understand the query
    query_analysis = QueryUnderstandingTool(query)
    
    # Determine if visualization is needed
    should_plot = query_analysis.get("visualization_type", "none") != "none"
    
    # Choose the appropriate tool based on whether visualization is needed
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query, query_analysis) if should_plot else CodeWritingTool(df.columns.tolist(), query, query_analysis)
    
    # Generate code using the LLM
    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code using modern libraries like pandas, numpy, scikit-learn, and plotly. Solve the given problem with optimal operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    
    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    
    return code, should_plot, query_analysis

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool, query_analysis: Dict[str, Any]):
    """Executes the generated code in a controlled environment and returns the result or visualization."""
    env = {
        "pd": pd, 
        "df": df, 
        "np": np, 
        "sns": sns,
        "StandardScaler": StandardScaler,
        "PCA": PCA
    }
    
    if should_plot:
        env["plt"] = plt
        env["px"] = px
        env["go"] = go
        env["io"] = io
        env["alt"] = alt
        
    try:
        exec(code, env)
        result = env.get("result", None)
        
        # If code execution didn't create a result, use the VisualizationTool as fallback
        if result is None and should_plot:
            viz_result = VisualizationTool(df, query_analysis)
            return viz_result
            
        # If result is a plotly figure, return as is
        if 'plotly.graph_objs._figure.Figure' in str(type(result)):
            return {"type": query_analysis.get("visualization_type", "custom"), 
                    "figure": result, 
                    "fig_type": "plotly"}
            
        # If result is a matplotlib figure, convert to plotly
        if isinstance(result, plt.Figure):
            return {"type": query_analysis.get("visualization_type", "custom"), 
                    "figure": result, 
                    "fig_type": "matplotlib"}
            
        # If result is an Altair chart
        if 'altair.vegalite' in str(type(result)):
            return {"type": query_analysis.get("visualization_type", "custom"), 
                    "figure": result, 
                    "fig_type": "altair"}
            
        # For other types of results (DataFrame, Series, scalar)
        return {"type": "table", 
                "data": result if isinstance(result, (pd.DataFrame, pd.Series)) else None,
                "scalar": result if not isinstance(result, (pd.DataFrame, pd.Series)) else None,
                "fig_type": "data"}
            
    except Exception as exc:
        st.error(f"Error executing code: {exc}")
        # Attempt to visualize with built-in tool as fallback
        try:
            viz_result = VisualizationTool(df, query_analysis)
            return viz_result
        except:
            return {"type": "error", "error": str(exc), "fig_type": "error"}

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the visualization or data result."""
    result_type = result.get("type", "unknown") if isinstance(result, dict) else "unknown"
    fig_type = result.get("fig_type", "unknown") if isinstance(result, dict) else "unknown"
    
    # Generate appropriate prompt based on result type
    if fig_type == "error":
        desc = result.get("error", "Unknown error occurred")
        prompt = f'''
        The user asked: "{query}".
        An error occurred: {desc}
        
        Please explain in 2–3 concise sentences what might have gone wrong and suggest an alternative approach.
        '''
    elif fig_type in ["plotly", "matplotlib", "altair"]:
        # Visualization result
        viz_type = result_type
        prompt = f'''
        The user asked: "{query}".
        
        I've created a {viz_type} visualization to answer this query.
        
        Please explain in 2–3 concise sentences:
        1. What insights this visualization reveals
        2. Any patterns, trends, or notable observations
        3. How this directly answers the user's question
        
        Keep your explanation data-focused and insightful without mentioning code details.
        '''
    elif fig_type == "table":
        # Table data result
        if "data" in result and result["data"] is not None:
            data_sample = str(result["data"].head(3)) if hasattr(result["data"], "head") else str(result["data"])[:500]
            prompt = f'''
            The user asked: "{query}".
            
            The result is a tabular dataset. Here's a sample:
            {data_sample}
            
            Please explain in 2–3 concise sentences what this table shows and how it answers the user's question.
            Focus on insights rather than simply describing what's in the table.
            '''
        else:
            prompt = f'''
            The user asked: "{query}".
            
            The result appears to be empty or cannot be displayed as a table.
            
            Please explain why this might be the case and suggest a different approach.
            '''
    elif fig_type == "data":
        # Scalar or other data result
        if "scalar" in result and result["scalar"] is not None:
            value = str(result["scalar"])
            prompt = f'''
            The user asked: "{query}".
            
            The result is: {value}
            
            Please explain in 1-2 concise sentences what this value means in context and how it answers the user's question.
            '''
        else:
            prompt = f'''
            The user asked: "{query}".
            
            No clear result was found. 
            
            Please explain why this might be the case and suggest what the user could try instead.
            '''
    else:
        # Unknown result type
        prompt = f'''
        The user asked: "{query}".
        
        I'm unable to determine the type of result. 
        
        Please suggest a different approach or query that might yield better results.
        '''
    
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result and extracts model 'thinking' and final explanation."""
    prompt = ReasoningCurator(query, result)
    
    # Streaming LLM call
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst providing clear, concise explanations of data visualizations and results."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )
    
    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            
            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )
    
    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# === DataFrameSummary TOOL =========================================
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a comprehensive summary prompt string for the LLM based on the DataFrame."""
    # Get basic dataframe info
    rows, cols = df.shape
    missing_data = df.isnull().sum().sum()
    missing_pct = (missing_data / (rows * cols)) * 100
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Attempt to identify date columns that might be stored as strings
    potential_date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dropna().shape[0] > 0:
            sample = df[col].dropna().iloc[0]
            try:
                pd.to_datetime(sample)
                potential_date_cols.append(col)
            except:
                pass
    
    # Get statistics for numeric columns
    numeric_stats = {}
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe().to_dict()
    
    # Get value counts for categorical columns (limited to top 5)
    categorical_stats = {}
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        if df[col].nunique() <= 10:  # Only for columns with reasonable number of categories
            categorical_stats[col] = df[col].value_counts().head(5).to_dict()
    
    # Detect correlations between numeric columns
    correlations = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if corr_value > 0.7:  # Only strong correlations
                    correlations[f"{col1} - {col2}"] = round(corr_value, 2)
    
    # Build the prompt
    prompt = f"""
        Based on analyzing a dataset with {rows} rows and {cols} columns:
        
        Dataset Overview:
        - Column names: {', '.join(df.columns)}
        - {len(numeric_cols)} numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
        - {len(categorical_cols)} categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
        - {len(datetime_cols)} datetime columns: {', '.join(datetime_cols) if datetime_cols else 'None'}
        - {len(potential_date_cols)} potential date columns (stored as strings): {', '.join(potential_date_cols) if potential_date_cols else 'None'}
        - Missing data: {missing_pct:.1f}% of all values
        
        Please provide:
        1. A clear, concise description of what this dataset contains and what it might be used for.
        2. The 3-4 most valuable insights someone could extract from this data.
        3. Suggested analyses or visualizations that would help understand this data better.
        4. Example questions a user could ask about this dataset.
        
        Keep your response structured, informative, and business-focused.
    """
    
    return prompt

# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(df: pd.DataFrame) -> Dict[str, Any]:
    """Uses the LLM to generate comprehensive insights about the uploaded dataset."""
    prompt = DataFrameSummaryTool(df)
    
    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a data analyst providing clear, structured insights about datasets. Format your response with markdown headings and bullet points for readability."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        
        insight_text = response.choices[0].message.content
        
        # Create initial visualizations
        initial_vizs = []
        
        # 1. Try to create a correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr()
                fig = px.imshow(
                    corr, 
                    title="Correlation Heatmap",
                    height=400,
                    color_continuous_scale="Viridis"
                )
                initial_vizs.append({
                    "title": "Correlation Heatmap",
                    "description": "Shows relationships between numeric variables",
                    "figure": fig,
                    "fig_type": "plotly"
                })
            except:
                pass
        
        # 2. Try to create a missing values heatmap
        try:
            missing = df.isnull()
            if missing.values.any():
                fig = px.imshow(
                    missing.transpose(),
                    title="Missing Values",
                    height=400,
                    color_continuous_scale=["white", "red"]
                )
                initial_vizs.append({
                    "title": "Missing Values",
                    "description": "Shows patterns in missing data",
                    "figure": fig,
                    "fig_type": "plotly"
                })
        except:
            pass
        
        # 3. For categorical columns, create a count chart for the first one
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and df[categorical_cols[0]].nunique() <= 20:
            try:
                col = categorical_cols[0]
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(
                    counts, x=col, y='count',
                    title=f"Distribution of {col}",
                    height=400
                )
                initial_vizs.append({
                    "title": f"Distribution of {col}",
                    "description": f"Shows counts for each category in {col}",
                    "figure": fig,
                    "fig_type": "plotly"
                })
            except:
                pass
        
        # 4. For numeric columns, create a histogram for the first one
        if len(numeric_cols) > 0:
            try:
                col = numeric_cols[0]
                fig = px.histogram(
                    df, x=col,
                    title=f"Distribution of {col}",
                    height=400
                )
                initial_vizs.append({
                    "title": f"Distribution of {col}",
                    "description": f"Shows the frequency distribution of {col}",
                    "figure": fig,
                    "fig_type": "plotly"
                })
            except:
                pass
        
        # Return both text insights and visualizations
        return {
            "text": insight_text,
            "visualizations": initial_vizs
        }
        
    except Exception as exc:
        return {
            "text": f"Error generating dataset insights: {exc}",
            "visualizations": []
        }

# === Visualization Helpers ===============================================

def get_plotly_config():
    """Returns configuration for Plotly charts to enable interactive features."""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'data_visualization',
            'height': 500,
            'width': 700,
            'scale': 2
        }
    }

def display_visualization(viz_result: Dict[str, Any], container=None):
    """Display a visualization result in a Streamlit container."""
    if container is None or container == st:
        # If no container or using the st module directly, use the default st
        display_container = st
    else:
        display_container = container
        
    fig_type = viz_result.get("fig_type", "unknown")
    
    if fig_type == "plotly" and viz_result.get("figure") is not None:
        # Display Plotly figure
        display_container.plotly_chart(viz_result["figure"], use_container_width=True, config=get_plotly_config())
            
    elif fig_type == "matplotlib" and viz_result.get("figure") is not None:
        # Display Matplotlib figure
        display_container.pyplot(viz_result["figure"], use_container_width=True)
            
    elif fig_type == "altair" and viz_result.get("figure") is not None:
        # Display Altair chart
        display_container.altair_chart(viz_result["figure"], use_container_width=True)
            
    elif fig_type == "table" and viz_result.get("data") is not None:
        # Display table data
        if isinstance(viz_result["data"], pd.DataFrame):
            display_container.dataframe(viz_result["data"], use_container_width=True)
        elif isinstance(viz_result["data"], pd.Series):
            display_container.dataframe(viz_result["data"].reset_index(), use_container_width=True)
            
    elif fig_type == "data" and viz_result.get("scalar") is not None:
        # Display scalar value
        display_container.metric("Result", viz_result["scalar"])
            
    elif fig_type == "error":
        # Display error message
        display_container.error(viz_result.get("error", "An unknown error occurred"))
            
    else:
        # Fallback for unknown visualization types
        display_container.warning("Unable to display the result. Try a different query.")

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""

# === Main Streamlit App ===============================================

def main():
    # Load custom CSS
    load_custom_css()
    
    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "visualizations" not in st.session_state:
        st.session_state.visualizations = []
    if "metadata" not in st.session_state:
        st.session_state.metadata = {}
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    # --- SIDEBAR: Upload, Preview, Insights ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/NVIDIA_logo_black.svg", width=150)
        st.title("Data Analysis Agent")
        st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</a></medium>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Data File", 
            type=["csv", "xlsx", "xls"],
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        
        if uploaded_file:
            # Save uploaded file to temp location
            file_path = os.path.join(".", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process the data if it's a new file
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != uploaded_file.name):
                with st.spinner("Analyzing your data..."):
                    # Load data and get metadata
                    df, metadata = DataLoaderTool(file_path)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.visualizations = []
                    
                    # Generate dataset insights
                    st.session_state.insights = DataInsightAgent(df)
                    
                    # Reset file uploader key to clear the uploader
                    st.session_state.file_uploader_key += 1
            
            # Display data preview
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.df.head(5), use_container_width=True)
            
            # Display dataset summary
            with st.expander("Dataset Summary", expanded=False):
                st.markdown(f"**Rows:** {st.session_state.metadata.get('rows', 0)}")
                st.markdown(f"**Columns:** {st.session_state.metadata.get('columns', 0)}")
                
                # Show column types
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Numeric Columns:**")
                    st.write(", ".join(st.session_state.metadata.get("numeric_columns", [])) or "None")
                with col2:
                    st.markdown("**Categorical Columns:**")
                    st.write(", ".join(st.session_state.metadata.get("categorical_columns", [])) or "None")
                
                # Show missing values
                st.markdown("**Missing Values:**")
                missing_cols = {k: v for k, v in st.session_state.metadata.get("missing_values", {}).items() if v > 0}
                if missing_cols:
                    st.write(missing_cols)
                else:
                    st.write("No missing values")
            
            # Display dataset insights
            if "insights" in st.session_state:
                st.markdown("### Dataset Insights")
                st.markdown(st.session_state.insights.get("text", "No insights available"))
                
                # Show initial visualizations in the sidebar
                if st.session_state.insights.get("visualizations", []):
                    st.markdown("### Quick Visualizations")
                    for viz in st.session_state.insights.get("visualizations", [])[:2]:  # Limit to 2 in sidebar
                        st.markdown(f"**{viz['title']}**")
                        st.markdown(viz["description"])
                        display_visualization(viz)
        else:
            # Show instructions when no file is uploaded
            st.info("👆 Upload a CSV or Excel file to begin analyzing your data.")
            st.markdown("""
            ### Welcome to Data Analysis Agent!
            
            This intelligent agent helps you analyze your data through natural language queries. Simply:
            
            1. **Upload** your data file (.csv or .xlsx)
            2. **Ask questions** in plain English
            3. **Get insights** with professional visualizations
            
            No coding required! Just chat with your data.
            """)
    
    # --- MAIN CONTENT ---
    if uploaded_file and "df" in st.session_state:
        # Main header with dynamic title based on file
        st.markdown(f"""
        <div class="main-header">
            📊 Analyzing: {st.session_state.current_file}
        </div>
        """, unsafe_allow_html=True)
        
        # --- CHAT INTERFACE ---
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
                    
                    # Display visualization if attached to message
                    if message.get("visualization") is not None:
                        display_visualization(message["visualization"])
        
        # Chat input
        query = st.chat_input("Ask about your data...")
        
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Create a placeholder for assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # Process the query
                    query_analysis = QueryUnderstandingTool(query)
                    
                    # Process data based on query
                    processed_df = DataProcessorTool(st.session_state.df, st.session_state.metadata, query_analysis)
                    
                    # Try direct visualization first
                    viz_result = VisualizationTool(processed_df, query_analysis)
                    
                    # If direct visualization failed or isn't appropriate, try code generation
                    if viz_result["fig_type"] == "error" or (
                        viz_result["fig_type"] == "table" and query_analysis["visualization_type"] != "table"
                    ):
                        # Generate and execute code
                        code, should_plot_flag, _ = CodeGenerationAgent(query, processed_df)
                        viz_result = ExecutionAgent(code, processed_df, should_plot_flag, query_analysis)
                    
                    # Generate reasoning about the result
                    thinking, reasoning = ReasoningAgent(query, viz_result)
                    
                    # Display reasoning
                    st.markdown(reasoning)
                    
                    # Display visualization
                    display_visualization(viz_result)
                    
                    # Store code for display if requested
                    viz_result["code"] = code if 'code' in locals() else ""
                    
                    # Add visualization to history and attach to message
                    st.session_state.visualizations.append(viz_result)
                    
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": reasoning,
                "visualization": viz_result
            })
            
            st.rerun()
            
        # --- VISUALIZATION GALLERY ---
        if st.session_state.visualizations:
            st.markdown("---")
            st.markdown("""
            <div class="sub-header">
                📈 Visualization Gallery
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different categories
            viz_types = list(set(viz["type"] for viz in st.session_state.visualizations if "type" in viz))
            if viz_types:
                tabs = st.tabs(["All Visualizations"] + viz_types)
                
                # All visualizations tab
                with tabs[0]:
                    cols = st.columns(2)
                    for i, viz in enumerate(reversed(st.session_state.visualizations)):
                        with cols[i % 2]:
                            with st.expander(f"Visualization {len(st.session_state.visualizations) - i}", expanded=i < 2):
                                display_visualization(viz)
                                if "code" in viz and viz["code"]:
                                    with st.expander("View Code"):
                                        st.code(viz["code"], language="python")
                
                # Tabs for each visualization type
                for j, viz_type in enumerate(viz_types):
                    with tabs[j+1]:
                        type_vizs = [viz for viz in st.session_state.visualizations if viz.get("type") == viz_type]
                        if type_vizs:
                            cols = st.columns(2)
                            for i, viz in enumerate(reversed(type_vizs)):
                                with cols[i % 2]:
                                    with st.expander(f"{viz_type.capitalize()} {len(type_vizs) - i}", expanded=i < 2):
                                        display_visualization(viz)
                        else:
                            st.info(f"No {viz_type} visualizations yet.")
    else:
        # Landing page when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>📊 Welcome to Data Analysis Agent</h1>
            <p style="font-size: 1.2rem; margin: 1.5rem 0;">
                Your intelligent assistant for data visualization and analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 💬 Natural Language
            Ask questions about your data in plain English - no coding required!
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Intelligent Visualization
            Automatically generates the most appropriate charts and graphs for your queries.
            """)
            
        with col3:
            st.markdown("""
            ### 🧠 Data Insights
            Get explanations and insights about your data, not just visualizations.
            """)
        
        # Example queries to try
        st.markdown("### Example Queries to Try")
        
        example_queries = [
            "Show me the distribution of sales by region",
            "What's the trend of revenue over time?",
            "Compare product categories by profit margin",
            "Show correlations between price and customer ratings",
            "Visualize the geographic distribution of customers",
            "What's the relationship between marketing spend and sales?"
        ]
        
        example_cols = st.columns(3)
        for i, query in enumerate(example_queries):
            with example_cols[i % 3]:
                st.markdown(f"- {query}")
        
        # Upload prompt
        st.info("👈 Upload your data file in the sidebar to get started!")


if __name__ == "__main__":
    main() 
