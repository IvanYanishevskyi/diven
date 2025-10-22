import pandas as pd
import io
from pathlib import Path

def detect_date_columns(df: pd.DataFrame) -> pd.DataFrame:

    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':  
            sample_values = df_copy[col].dropna().head(10)
            if len(sample_values) == 0:
                continue
                
            date_patterns = 0
            for value in sample_values:
                value_str = str(value).strip()
                if not value_str:
                    continue
                    
                date_indicators = [
                    '-' in value_str and len(value_str.split('-')) >= 2,  
                    '/' in value_str and len(value_str.split('/')) >= 2,  
                    value_str.count(':') in [1, 2], 
                    any(month in value_str.lower() for month in [
                        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
                    ])
                ]
                
                if any(date_indicators):
                    try:
                        pd.to_datetime(value_str)
                        date_patterns += 1
                    except (ValueError, TypeError):
                        continue
            
            if date_patterns / len(sample_values) > 0.5:
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    print(f"Converted column '{col}' to datetime")
                except Exception as e:
                    print(f"Failed to convert column '{col}' to datetime: {e}")
    
    return df_copy

def read_any_table(raw_bytes: bytes, filename: str = "") -> pd.DataFrame:

    if filename.lower().endswith('.csv'):
        df = pd.read_csv(io.BytesIO(raw_bytes),encoding="ISO-8859-1")
    elif filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(raw_bytes),encoding="ISO-8859-1")
    elif filename.lower().endswith('.json'):
        df = pd.read_json(io.BytesIO(raw_bytes),encoding="ISO-8859-1")
    elif filename.lower().endswith('.parquet'):
        df = pd.read_parquet(io.BytesIO(raw_bytes),encoding="ISO-8859-1")
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes),encoding="ISO-8859-1")
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes))
            except Exception:
                try:
                    df = pd.read_json(io.BytesIO(raw_bytes))
                except Exception as e:
                    raise ValueError(f"Cannot detect file format for {filename}: {e}")
    
    df = detect_date_columns(df)
    
    df = df.dropna(how='all') 
    df = df.loc[:, ~df.columns.duplicated()]  
    
    if len(df) == 0:
        raise ValueError("File is empty or contains no valid data")
    
    return df