#!/usr/bin/env python3
"""
ER Diagram Generator for Greyhound Racing Database
Generates a comprehensive Entity-Relationship diagram from the live database schema
"""

import re
import sqlite3
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def get_database_schema(db_path):
    """Extract complete schema information from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables (excluding backup tables for clarity)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%backup%' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {}

    for table in tables:
        table_info = {"columns": [], "foreign_keys": [], "indexes": [], "row_count": 0}

        # Get column information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        for col in columns:
            column_info = {
                "name": col[1],
                "type": col[2],
                "not_null": col[3] == 1,
                "default": col[4],
                "primary_key": col[5] == 1,
            }
            table_info["columns"].append(column_info)

        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()

        for fk in foreign_keys:
            fk_info = {
                "column": fk[3],
                "references_table": fk[2],
                "references_column": fk[4],
            }
            table_info["foreign_keys"].append(fk_info)

        # Get index information
        cursor.execute(f"PRAGMA index_list({table})")
        indexes = cursor.fetchall()

        for idx in indexes:
            if not idx[1].startswith("sqlite_autoindex"):  # Skip auto-generated indexes
                index_info = {"name": idx[1], "unique": idx[2] == 1}
                table_info["indexes"].append(index_info)

        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_info["row_count"] = cursor.fetchone()[0]
        except:
            table_info["row_count"] = 0

        schema_info[table] = table_info

    conn.close()
    return schema_info


def create_er_diagram(schema_info, output_path):
    """Create a comprehensive ER diagram"""

    # Create figure with larger size for readability
    fig, ax = plt.subplots(1, 1, figsize=(24, 20))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 20)
    ax.axis("off")

    # Define colors for different table types
    colors = {
        "core": "#E3F2FD",  # Light blue for core tables
        "lookup": "#F3E5F5",  # Light purple for lookup tables
        "analytics": "#E8F5E8",  # Light green for analytics tables
        "weather": "#FFF3E0",  # Light orange for weather tables
        "odds": "#FFEBEE",  # Light red for odds/betting tables
        "enhanced": "#F1F8E9",  # Light lime for enhanced/extra tables
    }

    # Categorize tables
    def categorize_table(table_name):
        if table_name in ["race_metadata", "dog_race_data", "dogs"]:
            return "core"
        elif table_name in ["venue_mappings", "trainers", "alembic_version"]:
            return "lookup"
        elif "weather" in table_name:
            return "weather"
        elif (
            "odds" in table_name
            or "value_bets" in table_name
            or "predictions" in table_name
        ):
            return "odds"
        elif "analytics" in table_name or "gpt_analysis" in table_name:
            return "analytics"
        elif (
            "ft_extra" in table_name
            or "gr_" in table_name
            or "enhanced" in table_name
            or "comprehensive" in table_name
        ):
            return "enhanced"
        else:
            return "core"

    # Position tables in a grid layout with logical grouping
    positions = {}

    # Core tables (center)
    core_tables = [t for t in schema_info.keys() if categorize_table(t) == "core"]
    for i, table in enumerate(core_tables):
        positions[table] = (11 + (i % 3) * 4, 14 - (i // 3) * 3)

    # Lookup tables (left side)
    lookup_tables = [t for t in schema_info.keys() if categorize_table(t) == "lookup"]
    for i, table in enumerate(lookup_tables):
        positions[table] = (2, 18 - i * 3)

    # Analytics tables (right side)
    analytics_tables = [
        t for t in schema_info.keys() if categorize_table(t) == "analytics"
    ]
    for i, table in enumerate(analytics_tables):
        positions[table] = (20, 18 - i * 3)

    # Weather tables (bottom left)
    weather_tables = [t for t in schema_info.keys() if categorize_table(t) == "weather"]
    for i, table in enumerate(weather_tables):
        positions[table] = (2 + (i % 2) * 5, 6 - (i // 2) * 3)

    # Odds tables (bottom right)
    odds_tables = [t for t in schema_info.keys() if categorize_table(t) == "odds"]
    for i, table in enumerate(odds_tables):
        positions[table] = (18 + (i % 2) * 3, 6 - (i // 2) * 3)

    # Enhanced tables (top)
    enhanced_tables = [
        t for t in schema_info.keys() if categorize_table(t) == "enhanced"
    ]
    for i, table in enumerate(enhanced_tables):
        positions[table] = (4 + (i % 4) * 4, 1 + (i // 4) * 2)

    # Draw tables
    table_boxes = {}
    for table_name, table_info in schema_info.items():
        if table_name not in positions:
            continue

        x, y = positions[table_name]

        # Calculate box height based on number of columns (max 8 visible)
        visible_columns = min(len(table_info["columns"]), 8)
        box_height = 0.3 + visible_columns * 0.15
        box_width = 3.5

        # Determine table color
        table_type = categorize_table(table_name)
        color = colors.get(table_type, colors["core"])

        # Create table box
        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(box)
        table_boxes[table_name] = (x, y, box_width, box_height)

        # Add table name (bold)
        ax.text(
            x,
            y + box_height / 2 - 0.15,
            table_name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

        # Add row count
        ax.text(
            x,
            y + box_height / 2 - 0.3,
            f"({table_info['row_count']:,} rows)",
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color="gray",
        )

        # Add key columns (show only first 6 to avoid clutter)
        key_columns = []
        for col in table_info["columns"][:6]:
            prefix = "🔑 " if col["primary_key"] else "• "
            key_columns.append(f"{prefix}{col['name']} ({col['type']})")

        if len(table_info["columns"]) > 6:
            key_columns.append(f"... and {len(table_info['columns']) - 6} more")

        for i, col_text in enumerate(key_columns):
            ax.text(
                x,
                y + box_height / 2 - 0.5 - i * 0.12,
                col_text,
                ha="center",
                va="center",
                fontsize=6,
            )

    # Draw foreign key relationships
    for table_name, table_info in schema_info.items():
        if table_name not in table_boxes:
            continue

        x1, y1, w1, h1 = table_boxes[table_name]

        for fk in table_info["foreign_keys"]:
            ref_table = fk["references_table"]
            if ref_table in table_boxes:
                x2, y2, w2, h2 = table_boxes[ref_table]

                # Draw relationship line
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.7),
                )

                # Add FK label
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{fk['column']}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

    # Add title and legend
    ax.text(
        12,
        19.5,
        "Greyhound Racing Database - Entity Relationship Diagram",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        12,
        19,
        f"Total Tables: {len(schema_info)} | Generated from Live Database",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
    )

    # Add legend
    legend_x = 0.5
    legend_y = 19
    ax.text(legend_x, legend_y, "Table Categories:", fontsize=10, fontweight="bold")

    categories = [
        ("Core Data", colors["core"]),
        ("Enhanced/Extra", colors["enhanced"]),
        ("Analytics", colors["analytics"]),
        ("Weather", colors["weather"]),
        ("Odds/Betting", colors["odds"]),
        ("Lookup", colors["lookup"]),
    ]

    for i, (category, color) in enumerate(categories):
        y_pos = legend_y - 0.4 - i * 0.3
        legend_box = patches.Rectangle(
            (legend_x, y_pos - 0.1), 0.3, 0.2, facecolor=color, edgecolor="black"
        )
        ax.add_patch(legend_box)
        ax.text(legend_x + 0.4, y_pos, category, fontsize=8, va="center")

    # Add relationship legend
    ax.text(legend_x, legend_y - 2.5, "Relationships:", fontsize=10, fontweight="bold")
    ax.annotate(
        "",
        xy=(legend_x + 0.8, legend_y - 2.8),
        xytext=(legend_x, legend_y - 2.8),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )
    ax.text(legend_x + 1, legend_y - 2.8, "Foreign Key", fontsize=8, va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"ER Diagram saved to: {output_path}")

    return fig


def main():
    db_path = "/Users/orlandolee/greyhound_racing_collector/greyhound_racing_data.db"
    output_path = "/Users/orlandolee/greyhound_racing_collector/database_er_diagram.png"

    print("🔍 Analyzing database schema...")
    schema_info = get_database_schema(db_path)

    print(f"📊 Found {len(schema_info)} tables")

    print("🎨 Generating ER diagram...")
    create_er_diagram(schema_info, output_path)

    print("✅ ER diagram generation complete!")


if __name__ == "__main__":
    main()
