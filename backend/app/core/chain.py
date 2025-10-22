import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from .prompts import prompt
from .schemas import Plan, AnalysisResponse
from .sandbox import exec_pandas, render_chart_png
from dotenv import load_dotenv
import traceback
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':  
            sample_values = df_processed[col].dropna().head(10)
            if len(sample_values) > 0:
                date_like_count = 0
                for val in sample_values:
                    try:
                        pd.to_datetime(str(val))
                        date_like_count += 1
                    except:
                        continue
                
                if date_like_count / len(sample_values) >= 0.5:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            df_processed[col] = pd.to_datetime(
                                df_processed[col], 
                                errors='coerce',
                                infer_datetime_format=True,
                                utc=False 
                            )
                        logger.info(f"Converted column '{col}' to datetime")
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
    
    return df_processed
def pick_example_cols(df: pd.DataFrame):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    col_num = num_cols[0] if num_cols else ""
    col_cat = cat_cols[0] if cat_cols else ""
    return col_cat, col_num
def profile_df(df: pd.DataFrame) -> str:
    type_info = []
    numeric_cols = []
    date_cols = []
    categorical_cols = []
    
    for c in df.columns:
        dtype_str = str(df[c].dtype)
        
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            if not df[c].isna().all():
                min_val, max_val = df[c].min(), df[c].max()
                type_info.append(f"{c}(numeric: {min_val:.2f} to {max_val:.2f})")
            else:
                type_info.append(f"{c}(numeric: all NaN)")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            if not df[c].isna().all():
                date_range = f"from {df[c].min()} to {df[c].max()}"
                type_info.append(f"{c}(datetime: {date_range})")
            else:
                type_info.append(f"{c}(datetime: all NaN)")
        else:
            categorical_cols.append(c)
            unique_count = df[c].nunique()
            type_info.append(f"{c}(categorical: {unique_count} unique)")
    
    dtypes = ", ".join(type_info)
    sample = df.head(3).fillna("").astype(str).to_dict(orient="records")
    
    profile_parts = [
        f"DataFrame Shape: {len(df)} rows x {df.shape[1]} columns",
        "",
        f"AVAILABLE COLUMNS (use these exact names in your code):",
        f"All columns: {list(df.columns)}",
    ]
    
    if numeric_cols:
        profile_parts.append(f"Numeric columns: {numeric_cols}")
    if date_cols:
        profile_parts.append(f"Date columns: {date_cols}")
    if categorical_cols:
        profile_parts.append(f"Categorical columns: {categorical_cols}")
    
    profile_parts.extend([
        "",
        f"Column Details: {dtypes}",
        "",
        f"Sample Data (first 3 rows): {sample}"
    ])
    
    return "\n".join(profile_parts)

def make_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0)

_llm = None
_parser = PydanticOutputParser(pydantic_object=Plan)

def analyze_question(df: pd.DataFrame, question: str) -> dict:
    global _llm
    if _llm is None:
        _llm = make_llm()

    try:
        df_processed = preprocess_dataframe(df)
        logger.info(f"Processing DataFrame with shape: {df_processed.shape}")
        
        prof = profile_df(df_processed)
        logger.info(f"DataFrame profile created")
        format_instructions = _parser.get_format_instructions()

        cols = list(df_processed.columns) if hasattr(df_processed, "columns") else []
        col_cat, col_num = pick_example_cols(df_processed)
        msgs = prompt.format_messages(
            question=question,
            profile=prof,                 
            columns=cols or [],        
            col_cat=col_cat or "",
            col_num=col_num or "",
            format_instructions=format_instructions         
        )
        
        logger.info("Sending request to LLM...")
        raw = _llm.invoke(msgs)
        
        logger.info("Parsing LLM response...")
        try:
            plan: Plan = _parser.parse(raw.content)
        except Exception as parse_error:
            logger.error(f"Failed to parse LLM response: {parse_error}")
            logger.error(f"Raw LLM response: {raw.content}")
            raise ValueError(f"LLM returned invalid JSON: {parse_error}")

        logger.info(f"Generated pandas code: {plan.pandas_code}")

        try:
            result = exec_pandas(plan.pandas_code, df_processed)
            logger.info(f"Code executed successfully, result type: {type(result)}")
        except Exception as exec_error:
            logger.error(f"Code execution failed: {exec_error}")
            logger.error(f"Failed code:\n{plan.pandas_code}")
            raise exec_error
        try:
            
            chart_b64 = render_chart_png(result, plan.chart)
            logger.info(f"Chart rendered successfully")
        except Exception as chart_error:
            logger.warning(f"Chart rendering failed: {chart_error}")
            chart_b64 = ""  
        used_cols = [c for c in df_processed.columns if c in plan.pandas_code]

        resp = AnalysisResponse(
            question=question,
            intent=plan.intent,
            reasoning=plan.reasoning,
            chart=plan.chart,
            chart_png_base64=chart_b64,
            preview_rows=len(df_processed),
            preview_cols=df_processed.shape[1],
            used_columns=used_cols or None,
            pandas_code=plan.pandas_code,
            answer_hint=plan.answer_hint,
        )
        
        logger.info("Analysis completed successfully")
        return resp.model_dump()
    
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        error_msg = str(e)
        
        if "KeyError" in error_msg or "'col" in error_msg or "Column" in error_msg:
            import re
            column_match = re.search(r"'([^']+)'", error_msg)
            missing_column = column_match.group(1) if column_match else "unknown"
            available_columns = list(df.columns) if 'df' in locals() else []
            
            error_msg = (
                f"Generated code tried to access column '{missing_column}' which doesn't exist. "
                f"Available columns: {available_columns}. "
                "The AI should use only existing column names from the DataFrame."
            )
        elif "Disallowed syntax" in error_msg:
            if "For" in error_msg:
                error_msg = (
                    "Generated code contains loops which are discouraged. "
                    "The AI should use vectorized pandas operations instead. "
                    f"Original error: {error_msg}"
                )
            else:
                error_msg = f"Generated code uses prohibited operations: {error_msg}"
        elif "not supported between instances" in error_msg:
            error_msg = (
                "Data type comparison error (likely datetime vs string). "
                f"Original error: {error_msg}"
            )
        elif "invalid JSON" in error_msg.lower():
            error_msg = f"AI response parsing failed: {error_msg}"
        elif "DEEPSEEK_API_KEY" in error_msg:
            error_msg = "DeepSeek API key not configured"
        
        raise Exception(f"Analysis failed: {error_msg}")