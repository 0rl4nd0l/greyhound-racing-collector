#!/usr/bin/env python3
"""
Database Schema Diff and Migration Script Generator

This script performs a detailed comparison between the expected database schema, defined
in `initialize_database.py`, and the actual schema introspected from the live database.

It generates two key outputs:
1.  **db_schema_diff.md**: A Markdown report detailing all discrepancies, including:
    - Missing or extra tables and columns
    - Mismatches in data types, nullability, and primary keys
    - Incorrect or missing foreign key constraints and indexes

2.  **post_warp_fix.sql**: A non-destructive SQL migration script containing the necessary
    `ALTER TABLE` and `CREATE INDEX` statements to align the live database schema with
    the expected schema.

Key functionalities:
-   Loads the actual schema from `database_schema_info.json`.
-   Constructs the expected schema from the `CREATE TABLE` statements in
    `initialize_database.py`.
-   Performs a field-by-field comparison of each table and column.
-   Identifies missing foreign keys and indexes.
-   Generates human-readable diffs and actionable SQL commands.

This automated approach ensures database integrity, facilitates seamless migrations, and
provides clear documentation of schema changes, which is crucial for maintaining a stable,
full-stack Greyhound Analysis Predictor.
"""

import json
import re
from pathlib import Path


def parse_create_table(sql_statement):
    """Parse a CREATE TABLE statement to extract schema information"""
    table_name_match = re.search(
        r"CREATE TABLE IF NOT EXISTS (\w+)", sql_statement, re.IGNORECASE
    )
    if not table_name_match:
        raise ValueError("Could not find table name")

    table_name = table_name_match.group(1)
    columns = []
    foreign_keys = []
    pk_columns = []

    # Find primary keys defined at the table level
    pk_match = re.search(r"PRIMARY KEY\s*\((.*?)\)", sql_statement, re.IGNORECASE)
    if pk_match:
        pk_columns = [c.strip() for c in pk_match.group(1).split(",")]

    # Split statement into lines and process each one
    lines = [line.strip() for line in sql_statement.split("\n") if line.strip()]

    # Find the start and end of the column definitions
    start_idx = -1
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if "CREATE TABLE" in line.upper():
            start_idx = i
        elif line.strip() == ")" or line.strip().endswith(");"):
            end_idx = i
            break

    if start_idx == -1:
        return table_name, {
            "columns": columns,
            "foreign_keys": foreign_keys,
            "indexes": [],
        }

    for line in lines[start_idx + 1 : end_idx]:
        line = line.strip().rstrip(",")

        # Skip comments and empty lines
        if not line or line.startswith("--") or line.startswith("/*"):
            continue

        parts = line.split()
        if not parts:
            continue

        if parts[0].upper() == "FOREIGN":
            fk_match = re.search(
                r"FOREIGN KEY\s*\((.*?)\)\s*REFERENCES\s*(.*?)\s*\((.*?)\)",
                line,
                re.IGNORECASE,
            )
            if fk_match:
                foreign_keys.append(
                    {
                        "from": fk_match.group(1).strip(),
                        "table": fk_match.group(2).strip(),
                        "to": fk_match.group(3).strip(),
                    }
                )
        elif (
            parts[0].upper() not in ["PRIMARY", "UNIQUE", "CONSTRAINT"]
            and len(parts) >= 2
        ):
            col_name = parts[0].strip("`")
            col_type = parts[1].strip("`")

            # Skip invalid column names
            if col_name.upper() in ["CREATE", "TABLE", "--", "/*", "*/"]:
                continue

            is_pk = "PRIMARY KEY" in line.upper() or col_name in pk_columns
            is_notnull = "NOT NULL" in line.upper()

            columns.append(
                {"name": col_name, "type": col_type, "pk": is_pk, "notnull": is_notnull}
            )

    return table_name, {"columns": columns, "foreign_keys": foreign_keys, "indexes": []}


def parse_create_index(sql_statement):
    """Parse a CREATE INDEX statement"""
    match = re.search(
        r"CREATE INDEX IF NOT EXISTS (\w+) ON (\w+)\s*\((.*?)\)",
        sql_statement,
        re.IGNORECASE,
    )
    if match:
        return {
            "name": match.group(1),
            "table": match.group(2),
            "columns": [c.strip() for c in match.group(3).split(",")],
        }
    return None


def load_expected_schema_from_multiple_sources():
    """Load expected schema from multiple sources"""
    schema = {}

    # Source 1: initialize_database.py (triple quotes)
    init_content = Path("initialize_database.py").read_text()
    create_table_matches = re.findall(
        r"cursor\.execute\(\'\'\'\s*(.*?)\s*\'\'\'\)",
        init_content,
        re.DOTALL | re.IGNORECASE,
    )
    for match in create_table_matches:
        if "CREATE TABLE" in match.upper():
            try:
                table_name, table_info = parse_create_table(match.strip())
                schema[table_name] = table_info
            except Exception as e:
                print(f"Error parsing table from initialize_database.py: {e}")
                continue

    # Source 2: create_unified_database.py (executescript with multiline SQL)
    unified_content = Path("create_unified_database.py").read_text()
    unified_matches = re.findall(
        r'cursor\.executescript\("""(.*?)"""\)',
        unified_content,
        re.DOTALL | re.IGNORECASE,
    )
    for match in unified_matches:
        statements = re.split(r";\s*(?=CREATE|--)", match)
        for statement in statements:
            if "CREATE TABLE" in statement.upper():
                try:
                    table_name, table_info = parse_create_table(statement.strip())
                    schema[table_name] = table_info
                except Exception as e:
                    print(f"Error parsing table from create_unified_database.py: {e}")
                    continue

    # Source 3: create_tables.sql direct SQL file
    if Path("create_tables.sql").exists():
        sql_content = Path("create_tables.sql").read_text()
        sql_statements = re.split(r";\s*(?=--)|;\s*(?=CREATE)", sql_content)
        for statement in sql_statements:
            if "CREATE TABLE" in statement.upper():
                try:
                    table_name, table_info = parse_create_table(statement.strip())
                    schema[table_name] = table_info
                except Exception as e:
                    print(f"Error parsing table from create_tables.sql: {e}")
                    continue

    # Find CREATE INDEX statements from all sources
    all_sources = [init_content, unified_content]
    if Path("create_tables.sql").exists():
        all_sources.append(Path("create_tables.sql").read_text())

    for content in all_sources:
        create_index_matches = re.findall(r"CREATE INDEX.*?;", content, re.IGNORECASE)
        for match in create_index_matches:
            index_info = parse_create_index(match)
            if index_info and index_info["table"] in schema:
                schema[index_info["table"]]["indexes"].append(index_info)

    return schema


def compare_schemas(expected_schema, actual_schema):
    """Compare expected and actual schemas and return a diff"""
    diff = {"missing_tables": [], "extra_tables": [], "table_diffs": {}}

    expected_tables = set(expected_schema.keys())
    actual_tables = set(actual_schema["tables"].keys())

    diff["missing_tables"] = list(expected_tables - actual_tables)
    diff["extra_tables"] = list(actual_tables - expected_tables)

    for table_name in expected_tables.intersection(actual_tables):
        table_diff = {
            "missing_columns": [],
            "extra_columns": [],
            "column_mismatches": [],
            "missing_foreign_keys": [],
            "extra_foreign_keys": [],
            "missing_indexes": [],
        }

        expected_table = expected_schema[table_name]
        actual_table = actual_schema["tables"][table_name]

        # Compare columns
        expected_cols = {col["name"]: col for col in expected_table["columns"]}
        actual_cols = {col["name"]: col for col in actual_table["columns"]}

        missing_col_names = set(expected_cols.keys()) - set(actual_cols.keys())
        extra_col_names = set(actual_cols.keys()) - set(expected_cols.keys())

        table_diff["missing_columns"] = [
            expected_cols[name] for name in missing_col_names
        ]
        table_diff["extra_columns"] = [actual_cols[name] for name in extra_col_names]

        for col_name in set(expected_cols.keys()).intersection(set(actual_cols.keys())):
            expected_col = expected_cols[col_name]
            actual_col = actual_cols[col_name]
            mismatches = []

            # Compare types (normalize)
            expected_type = expected_col["type"].upper()
            actual_type = actual_col["type"].upper()
            if expected_type != actual_type and not (
                expected_type == "DECIMAL" and "DECIMAL" in actual_type
            ):
                mismatches.append(
                    f"type mismatch (expected: {expected_type}, actual: {actual_type})"
                )

            if expected_col["notnull"] != actual_col["notnull"]:
                mismatches.append(
                    f"nullability mismatch (expected: {expected_col['notnull']}, actual: {actual_col['notnull']})"
                )

            if expected_col["pk"] != actual_col["pk"]:
                mismatches.append(
                    f"PK mismatch (expected: {expected_col['pk']}, actual: {actual_col['pk']})"
                )

            if mismatches:
                table_diff["column_mismatches"].append(
                    {"column": col_name, "mismatches": mismatches}
                )

        # Compare foreign keys
        expected_fks = [
            (fk["from"], fk["table"], fk["to"]) for fk in expected_table["foreign_keys"]
        ]
        actual_fks = [
            (fk["from"], fk["table"], fk["to"]) for fk in actual_table["foreign_keys"]
        ]

        table_diff["missing_foreign_keys"] = [
            fk for fk in expected_fks if fk not in actual_fks
        ]
        table_diff["extra_foreign_keys"] = [
            fk for fk in actual_fks if fk not in expected_fks
        ]

        # Compare indexes
        expected_idxs = [
            (idx["name"], tuple(sorted(idx["columns"])))
            for idx in expected_table["indexes"]
        ]
        actual_idxs_raw = actual_table.get("indexes", [])
        actual_idxs = [
            (idx["name"], tuple(sorted([c["name"] for c in idx["columns"]])))
            for idx in actual_idxs_raw
            if not idx["name"].startswith("sqlite_autoindex")
        ]

        table_diff["missing_indexes"] = [
            idx for idx in expected_idxs if idx not in actual_idxs
        ]

        # Only add to diff if there are changes
        if any(table_diff.values()):
            diff["table_diffs"][table_name] = table_diff

    return diff


def generate_diff_report(diff, report_path):
    """Generate a Markdown report of the schema differences"""
    report = ["# Database Schema Difference Report"]

    if diff["missing_tables"]:
        report.append("## Missing Tables")
        report.extend([f"- `{table}`" for table in diff["missing_tables"]])

    if diff["extra_tables"]:
        report.append("## Extra Tables")
        report.extend([f"- `{table}`" for table in diff["extra_tables"]])

    if diff["table_diffs"]:
        report.append("## Table Differences")
        for table, table_diff in diff["table_diffs"].items():
            report.append(f"### Table: `{table}`")

            if table_diff["missing_columns"]:
                report.append("#### Missing Columns")
                for col in table_diff["missing_columns"]:
                    report.append(f"- `{col['name']}` ({col['type']})")

            if table_diff["extra_columns"]:
                report.append("#### Extra Columns")
                for col in table_diff["extra_columns"]:
                    report.append(f"- `{col['name']}` ({col['type']})")

            if table_diff["column_mismatches"]:
                report.append("#### Column Mismatches")
                for mismatch in table_diff["column_mismatches"]:
                    report.append(
                        f"- **`{mismatch['column']}`**: { ', '.join(mismatch['mismatches']) }"
                    )

            if table_diff["missing_foreign_keys"]:
                report.append("#### Missing Foreign Keys")
                for fk in table_diff["missing_foreign_keys"]:
                    report.append(f"- `{fk[0]}` -> `{fk[1]}({fk[2]})`")

            if table_diff["missing_indexes"]:
                report.append("#### Missing Indexes")
                for idx in table_diff["missing_indexes"]:
                    report.append(f"- Index `{idx[0]}` on `({ ', '.join(idx[1]) })`")

    Path(report_path).write_text("\n".join(report))
    print(f"Diff report saved to: {report_path}")


def generate_migration_sql(diff, sql_path):
    """Generate a non-destructive SQL migration script"""
    sql = ["-- Non-destructive database migration to fix schema mismatches"]

    # Note: SQLite has limited ALTER TABLE support. We can add columns and create indexes.
    # Modifying constraints or column types often requires recreating the table.

    for table, table_diff in diff["table_diffs"].items():
        if table_diff["missing_columns"]:
            sql.append(f"\n-- Add missing columns to {table}")
            for col in table_diff["missing_columns"]:
                sql.append(
                    f"ALTER TABLE {table} ADD COLUMN {col['name']} {col['type']};"
                )

        if table_diff["missing_indexes"]:
            sql.append(f"\n-- Create missing indexes for {table}")
            for idx in table_diff["missing_indexes"]:
                sql.append(
                    f"CREATE INDEX IF NOT EXISTS {idx[0]} ON {table} ({ ', '.join(idx[1]) });"
                )

    # Note about limitations
    sql.append(
        "\n-- NOTE: Modifying column types, constraints, or foreign keys in SQLite often requires recreating the table."
    )
    sql.append("-- These changes should be reviewed and applied manually if needed.")

    Path(sql_path).write_text("\n".join(sql))
    print(f"Migration SQL saved to: {sql_path}")


if __name__ == "__main__":
    # Load schemas
    actual_schema = json.loads(Path("database_schema_info.json").read_text())
    expected_schema = load_expected_schema_from_multiple_sources()

    print(
        f"Expected schema found {len(expected_schema)} tables: {list(expected_schema.keys())}"
    )
    print(f"Actual schema has {len(actual_schema['tables'])} tables")

    # Compare and generate outputs
    schema_diff = compare_schemas(expected_schema, actual_schema)
    generate_diff_report(schema_diff, "reports/db_schema_diff.md")
    generate_migration_sql(schema_diff, "migrations/post_warp_fix.sql")

    print("\nSchema validation complete.")
