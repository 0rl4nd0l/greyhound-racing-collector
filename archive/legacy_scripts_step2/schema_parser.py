
import pandas as pd
import re

def parse_schema_to_dataframe(schema):
    """
    Parses a SQL schema dump and returns a Pandas DataFrame of tables and columns.

    Args:
        schema (str): The SQL schema as a string.

    Returns:
        pd.DataFrame: A DataFrame with columns ['table_name', 'column_name', 'data_type'].
    """
    tables = []
    # Regex to find table name and columns
    table_regex = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)?\s+\"?(\w+)\"?\s*\((.*?)\);", re.DOTALL | re.IGNORECASE)
    column_regex = re.compile(r"^\s*\"?(\w+)\"?\s+([\w\(\)]+)", re.IGNORECASE)

    for match in table_regex.finditer(schema):
        table_name, columns_str = match.groups()
        # Clean up column string
        columns_str = re.sub(r'\/\*.*?\*\/', '', columns_str) # remove comments
        columns_str = re.sub(r'--.*', '', columns_str) # remove comments

        lines = [line.strip() for line in columns_str.split('\n') if line.strip()]
        
        for line in lines:
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]

            # Skip constraints
            if line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CONSTRAINT', 'CHECK')):
                continue
            
            # Get column name and type
            col_match = column_regex.match(line)
            if col_match:
                column_name, data_type = col_match.groups()
                tables.append({'table_name': table_name.strip(), 'column_name': column_name.strip(), 'data_type': data_type.strip()})

    return pd.DataFrame(tables)

if __name__ == '__main__':
    with open('schema_dump.txt', 'r') as f:
        schema_dump = f.read()
    
    df = parse_schema_to_dataframe(schema_dump)
    print(df.to_string())
