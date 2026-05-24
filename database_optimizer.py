#!/usr/bin/env python3
"""
Database Optimization and Maintenance Utility
==============================================

Comprehensive database optimization tool providing:
- Index creation and optimization
- Query performance analysis
- Data integrity checks
- Database maintenance operations
- Statistics and reporting

Author: AI Assistant
Date: August 23, 2025
"""

import argparse
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IndexInfo:
    """Information about a database index"""

    name: str
    table: str
    columns: List[str]
    unique: bool
    sql: str


@dataclass
class QueryPerformance:
    """Query performance metrics"""

    query: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    uses_index: bool
    suggested_indexes: List[str]


class DatabaseOptimizer:
    """Comprehensive database optimization and maintenance"""

    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path

        # Recommended indexes for optimal performance
        self.recommended_indexes = {
            "race_metadata": [
                ("idx_results_status", ["results_status"]),
                ("idx_winner_source", ["winner_source"]),
                ("idx_venue_date", ["venue", "race_date"]),
                ("idx_race_date", ["race_date"]),
                ("idx_scraping_attempts", ["scraping_attempts"]),
                ("idx_status_date", ["results_status", "race_date"]),
                ("idx_winner_name", ["winner_name"]),
                ("idx_extraction_timestamp", ["extraction_timestamp"]),
            ],
            "dog_race_data": [
                ("idx_race_id", ["race_id"]),
                ("idx_dog_clean_name", ["dog_clean_name"]),
                ("idx_finish_position", ["finish_position"]),
                ("idx_box_number", ["box_number"]),
                ("idx_trainer_name", ["trainer_name"]),
                ("idx_race_dog", ["race_id", "dog_clean_name"]),
            ],
            "race_analytics": [
                ("idx_race_analytics_race_id", ["race_id"]),
                ("idx_race_analytics_type", ["analysis_type"]),
                ("idx_race_analytics_timestamp", ["analysis_timestamp"]),
            ],
            "track_conditions": [
                ("idx_track_conditions_venue", ["venue"]),
                ("idx_track_conditions_date", ["date"]),
                ("idx_track_conditions_venue_date", ["venue", "date"]),
            ],
        }

    def analyze_current_indexes(self) -> Dict[str, List[IndexInfo]]:
        """Analyze current database indexes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            indexes_by_table = {}

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA index_list({table})")
                table_indexes = cursor.fetchall()

                indexes_by_table[table] = []

                for idx_row in table_indexes:
                    # Handle different SQLite versions that return different numbers of columns
                    if len(idx_row) >= 4:
                        idx_name, unique, origin, partial = idx_row[:4]
                    else:
                        idx_name, unique, origin = idx_row[:3]
                        partial = 0

                    # Get index info
                    cursor.execute(f"PRAGMA index_info({idx_name})")
                    idx_info = cursor.fetchall()
                    columns = [col[2] for col in idx_info]

                    # Get index SQL
                    cursor.execute(
                        "SELECT sql FROM sqlite_master WHERE name = ?", (idx_name,)
                    )
                    sql_result = cursor.fetchone()
                    sql = sql_result[0] if sql_result else "SYSTEM INDEX"

                    index_info = IndexInfo(
                        name=idx_name,
                        table=table,
                        columns=columns,
                        unique=bool(unique),
                        sql=sql,
                    )

                    indexes_by_table[table].append(index_info)

            return indexes_by_table

        finally:
            conn.close()

    def get_missing_indexes(self) -> Dict[str, List[Tuple[str, List[str]]]]:
        """Identify missing recommended indexes"""
        current_indexes = self.analyze_current_indexes()
        missing_indexes = {}

        for table, recommended in self.recommended_indexes.items():
            if table not in current_indexes:
                missing_indexes[table] = recommended
                continue

            current_index_columns = []
            for idx in current_indexes[table]:
                current_index_columns.append(tuple(idx.columns))

            missing_for_table = []
            for idx_name, columns in recommended:
                if tuple(columns) not in current_index_columns:
                    missing_for_table.append((idx_name, columns))

            if missing_for_table:
                missing_indexes[table] = missing_for_table

        return missing_indexes

    def create_missing_indexes(self, dry_run: bool = False) -> Dict[str, Any]:
        """Create missing recommended indexes"""
        missing = self.get_missing_indexes()

        if not missing:
            return {
                "status": "success",
                "message": "All recommended indexes exist",
                "created": [],
            }

        if dry_run:
            return {
                "status": "dry_run",
                "message": f"Would create {sum(len(indexes) for indexes in missing.values())} indexes",
                "missing_indexes": missing,
            }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        created_indexes = []
        errors = []

        try:
            for table, indexes in missing.items():
                for idx_name, columns in indexes:
                    try:
                        # Create index
                        columns_str = ", ".join(columns)
                        sql = f"CREATE INDEX {idx_name} ON {table} ({columns_str})"

                        print(f"Creating index: {idx_name} on {table}({columns_str})")
                        start_time = time.time()
                        cursor.execute(sql)
                        creation_time = time.time() - start_time

                        created_indexes.append(
                            {
                                "name": idx_name,
                                "table": table,
                                "columns": columns,
                                "creation_time_seconds": creation_time,
                                "sql": sql,
                            }
                        )

                        print(f"   ✅ Created in {creation_time:.2f}s")

                    except Exception as e:
                        error_msg = f"Failed to create {idx_name}: {str(e)}"
                        errors.append(error_msg)
                        print(f"   ❌ {error_msg}")

            conn.commit()

            return {
                "status": "success",
                "created": created_indexes,
                "errors": errors,
                "total_created": len(created_indexes),
                "total_errors": len(errors),
            }

        except Exception as e:
            conn.rollback()
            return {"status": "error", "error": str(e)}
        finally:
            conn.close()

    def analyze_query_performance(self, queries: List[str]) -> List[QueryPerformance]:
        """Analyze performance of specific queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []

        try:
            for query in queries:
                try:
                    # Get query plan
                    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                    query_plan = cursor.fetchall()

                    # Check if indexes are used
                    uses_index = any(
                        "USING INDEX" in str(row).upper() for row in query_plan
                    )

                    # Measure execution time
                    start_time = time.time()
                    cursor.execute(query)
                    results_data = cursor.fetchall()
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms

                    # Analyze query plan for suggestions
                    suggested_indexes = self._suggest_indexes_for_query(
                        query, query_plan
                    )

                    perf = QueryPerformance(
                        query=query,
                        execution_time_ms=execution_time,
                        rows_examined=len(results_data),  # Simplified metric
                        rows_returned=len(results_data),
                        uses_index=uses_index,
                        suggested_indexes=suggested_indexes,
                    )

                    results.append(perf)

                except Exception as e:
                    print(f"Error analyzing query: {str(e)}")
                    continue

            return results

        finally:
            conn.close()

    def _suggest_indexes_for_query(
        self, query: str, query_plan: List[Tuple]
    ) -> List[str]:
        """Suggest indexes based on query analysis"""
        suggestions = []

        # Simple heuristics for index suggestions
        query_upper = query.upper()

        # Look for WHERE clauses
        if "WHERE" in query_upper:
            # Common patterns that benefit from indexes
            if "RESULTS_STATUS" in query_upper:
                suggestions.append("results_status")
            if "RACE_DATE" in query_upper:
                suggestions.append("race_date")
            if "VENUE" in query_upper:
                suggestions.append("venue")
            if "WINNER_NAME" in query_upper:
                suggestions.append("winner_name")
            if "SCRAPING_ATTEMPTS" in query_upper:
                suggestions.append("scraping_attempts")

        # Look for JOIN operations
        if "JOIN" in query_upper:
            if "race_id" in query.lower():
                suggestions.append("race_id")

        # Look for ORDER BY
        if "ORDER BY" in query_upper:
            if "race_date" in query.lower():
                suggestions.append("race_date")
            if "extraction_timestamp" in query.lower():
                suggestions.append("extraction_timestamp")

        return list(set(suggestions))  # Remove duplicates

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            stats = {
                "tables": {},
                "indexes": {},
                "database_info": {},
                "performance_metrics": {},
            }

            # Database file info
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            stats["database_info"] = {
                "file_path": self.db_path,
                "page_count": page_count,
                "page_size": page_size,
                "database_size_mb": (page_count * page_size) / (1024 * 1024),
                "sqlite_version": sqlite3.sqlite_version,
            }

            # Table statistics
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]

                stats["tables"][table] = {"row_count": row_count, "indexes": []}

                # Index information for this table
                cursor.execute(f"PRAGMA index_list({table})")
                table_indexes = cursor.fetchall()

                for idx_row in table_indexes:
                    idx_name = idx_row[0]
                    stats["tables"][table]["indexes"].append(idx_name)

            # Performance metrics with common queries
            common_queries = [
                "SELECT COUNT(*) FROM race_metadata WHERE results_status = 'pending'",
                "SELECT COUNT(*) FROM race_metadata WHERE results_status = 'complete'",
                "SELECT venue, COUNT(*) FROM race_metadata GROUP BY venue LIMIT 10",
                "SELECT * FROM race_metadata ORDER BY race_date DESC LIMIT 10",
                "SELECT COUNT(*) FROM dog_race_data WHERE finish_position = '1'",
            ]

            performance_results = self.analyze_query_performance(common_queries)
            stats["performance_metrics"] = {
                "common_queries": [
                    {
                        "query": perf.query,
                        "execution_time_ms": perf.execution_time_ms,
                        "uses_index": perf.uses_index,
                    }
                    for perf in performance_results
                ]
            }

            return stats

        finally:
            conn.close()

    def check_data_integrity(self) -> Dict[str, Any]:
        """Perform data integrity checks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            integrity_report = {
                "foreign_key_violations": [],
                "data_consistency_issues": [],
                "missing_data_issues": [],
                "duplicate_issues": [],
                "status": "checking",
            }

            # Check foreign key constraints
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            if fk_violations:
                integrity_report["foreign_key_violations"] = [
                    {"table": row[0], "rowid": row[1], "parent": row[2], "fkid": row[3]}
                    for row in fk_violations
                ]

            # Check for races without dogs
            cursor.execute(
                """
                SELECT race_id FROM race_metadata 
                WHERE race_id NOT IN (SELECT DISTINCT race_id FROM dog_race_data)
            """
            )
            races_without_dogs = cursor.fetchall()
            if races_without_dogs:
                integrity_report["data_consistency_issues"].append(
                    {
                        "issue": "races_without_dogs",
                        "count": len(races_without_dogs),
                        "sample": [row[0] for row in races_without_dogs[:5]],
                    }
                )

            # Check for dogs without races
            cursor.execute(
                """
                SELECT DISTINCT race_id FROM dog_race_data 
                WHERE race_id NOT IN (SELECT race_id FROM race_metadata)
            """
            )
            orphaned_dogs = cursor.fetchall()
            if orphaned_dogs:
                integrity_report["data_consistency_issues"].append(
                    {
                        "issue": "orphaned_dog_records",
                        "count": len(orphaned_dogs),
                        "sample": [row[0] for row in orphaned_dogs[:5]],
                    }
                )

            # Check for missing winner names in complete races
            cursor.execute(
                """
                SELECT COUNT(*) FROM race_metadata 
                WHERE results_status = 'complete' 
                AND (winner_name IS NULL OR winner_name = '')
            """
            )
            missing_winners = cursor.fetchone()[0]
            if missing_winners > 0:
                integrity_report["missing_data_issues"].append(
                    {
                        "issue": "complete_races_missing_winners",
                        "count": missing_winners,
                    }
                )

            # Check for duplicate race IDs
            cursor.execute(
                """
                SELECT race_id, COUNT(*) as count FROM race_metadata 
                GROUP BY race_id HAVING COUNT(*) > 1
            """
            )
            duplicate_races = cursor.fetchall()
            if duplicate_races:
                integrity_report["duplicate_issues"].append(
                    {
                        "issue": "duplicate_race_ids",
                        "count": len(duplicate_races),
                        "sample": [
                            {"race_id": row[0], "duplicates": row[1]}
                            for row in duplicate_races[:5]
                        ],
                    }
                )

            # Overall status
            total_issues = (
                len(integrity_report["foreign_key_violations"])
                + len(integrity_report["data_consistency_issues"])
                + len(integrity_report["missing_data_issues"])
                + len(integrity_report["duplicate_issues"])
            )

            if total_issues == 0:
                integrity_report["status"] = "healthy"
            elif total_issues < 10:
                integrity_report["status"] = "minor_issues"
            else:
                integrity_report["status"] = "needs_attention"

            integrity_report["total_issues"] = total_issues

            return integrity_report

        finally:
            conn.close()

    def optimize_database(
        self, vacuum: bool = True, analyze: bool = True
    ) -> Dict[str, Any]:
        """Perform database optimization operations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = {
            "operations": [],
            "before_size_mb": 0,
            "after_size_mb": 0,
            "time_taken_seconds": 0,
        }

        start_time = time.time()

        try:
            # Get initial size
            cursor.execute("PRAGMA page_count")
            page_count_before = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            results["before_size_mb"] = (page_count_before * page_size) / (1024 * 1024)

            # VACUUM operation (reclaim unused space)
            if vacuum:
                print("🔧 Running VACUUM...")
                vacuum_start = time.time()
                cursor.execute("VACUUM")
                vacuum_time = time.time() - vacuum_start

                results["operations"].append(
                    {
                        "operation": "VACUUM",
                        "time_seconds": vacuum_time,
                        "description": "Reclaimed unused database space",
                    }
                )
                print(f"   ✅ VACUUM completed in {vacuum_time:.2f}s")

            # ANALYZE operation (update query planner statistics)
            if analyze:
                print("📊 Running ANALYZE...")
                analyze_start = time.time()
                cursor.execute("ANALYZE")
                analyze_time = time.time() - analyze_start

                results["operations"].append(
                    {
                        "operation": "ANALYZE",
                        "time_seconds": analyze_time,
                        "description": "Updated query planner statistics",
                    }
                )
                print(f"   ✅ ANALYZE completed in {analyze_time:.2f}s")

            # Get final size
            cursor.execute("PRAGMA page_count")
            page_count_after = cursor.fetchone()[0]
            results["after_size_mb"] = (page_count_after * page_size) / (1024 * 1024)
            results["space_saved_mb"] = (
                results["before_size_mb"] - results["after_size_mb"]
            )

            results["time_taken_seconds"] = time.time() - start_time

            return results

        finally:
            conn.close()

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        print("🔍 Analyzing database...")

        # Get database statistics
        stats = self.get_database_statistics()

        # Check for missing indexes
        missing_indexes = self.get_missing_indexes()

        # Check data integrity
        integrity = self.check_data_integrity()

        report = [
            "🏁 DATABASE OPTIMIZATION REPORT",
            "=" * 60,
            "",
            f"📊 DATABASE INFO:",
            f"   File: {stats['database_info']['file_path']}",
            f"   Size: {stats['database_info']['database_size_mb']:.2f} MB",
            f"   Pages: {stats['database_info']['page_count']:,}",
            f"   SQLite Version: {stats['database_info']['sqlite_version']}",
            "",
            f"📋 TABLE STATISTICS:",
        ]

        for table, table_stats in stats["tables"].items():
            report.append(
                f"   {table}: {table_stats['row_count']:,} rows, {len(table_stats['indexes'])} indexes"
            )

        # Index analysis
        report.extend(["", "🔍 INDEX ANALYSIS:"])
        if missing_indexes:
            report.append(
                f"   ⚠️  {sum(len(indexes) for indexes in missing_indexes.values())} recommended indexes missing"
            )
            for table, indexes in missing_indexes.items():
                report.append(f"   📋 {table}: {len(indexes)} missing indexes")
                for idx_name, columns in indexes:
                    report.append(f"      - {idx_name}: ({', '.join(columns)})")
        else:
            report.append("   ✅ All recommended indexes exist")

        # Performance metrics
        report.extend(["", "⚡ PERFORMANCE METRICS:"])
        if "common_queries" in stats["performance_metrics"]:
            slow_queries = [
                q
                for q in stats["performance_metrics"]["common_queries"]
                if q["execution_time_ms"] > 100
            ]
            if slow_queries:
                report.append(
                    f"   ⚠️  {len(slow_queries)} slow queries detected (>100ms)"
                )
                for query in slow_queries[:3]:
                    report.append(
                        f"      - {query['execution_time_ms']:.1f}ms: {query['query'][:50]}..."
                    )
            else:
                report.append("   ✅ All common queries perform well (<100ms)")

        # Data integrity
        report.extend(["", "🔒 DATA INTEGRITY:"])
        report.append(f"   Status: {integrity['status'].upper()}")
        if integrity["total_issues"] > 0:
            report.append(f"   Issues found: {integrity['total_issues']}")
            for issue_type in [
                "foreign_key_violations",
                "data_consistency_issues",
                "missing_data_issues",
                "duplicate_issues",
            ]:
                issues = integrity[issue_type]
                if issues:
                    report.append(
                        f"   - {issue_type.replace('_', ' ').title()}: {len(issues)}"
                    )
        else:
            report.append("   ✅ No integrity issues found")

        # Recommendations
        report.extend(["", "💡 RECOMMENDATIONS:"])

        if missing_indexes:
            report.append("   🔧 Create missing indexes:")
            report.append("      python3 database_optimizer.py create-indexes")

        if stats["database_info"]["database_size_mb"] > 100:
            report.append("   🗂️  Database is large - consider VACUUM:")
            report.append("      python3 database_optimizer.py optimize --vacuum")

        if integrity["total_issues"] > 0:
            report.append("   🔒 Fix data integrity issues:")
            report.append("      python3 database_optimizer.py integrity --fix")

        report.extend(
            [
                "",
                "🔧 OPTIMIZATION COMMANDS:",
                "   # Create missing indexes:",
                "   python3 database_optimizer.py create-indexes",
                "",
                "   # Optimize database:",
                "   python3 database_optimizer.py optimize --vacuum --analyze",
                "",
                "   # Full optimization:",
                "   python3 database_optimizer.py full-optimize",
            ]
        )

        return "\n".join(report)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Database Optimization Utility")
    parser.add_argument(
        "--db", default="greyhound_racing_data.db", help="Database path"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate optimization report")

    # Create indexes command
    indexes_parser = subparsers.add_parser(
        "create-indexes", help="Create missing indexes"
    )
    indexes_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be created"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize database")
    optimize_parser.add_argument("--vacuum", action="store_true", help="Run VACUUM")
    optimize_parser.add_argument("--analyze", action="store_true", help="Run ANALYZE")

    # Full optimize command
    full_optimize_parser = subparsers.add_parser(
        "full-optimize", help="Full optimization"
    )

    # Integrity command
    integrity_parser = subparsers.add_parser("integrity", help="Check data integrity")

    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    optimizer = DatabaseOptimizer(args.db)

    if args.command == "report":
        print(optimizer.generate_optimization_report())

    elif args.command == "create-indexes":
        result = optimizer.create_missing_indexes(dry_run=args.dry_run)

        if result["status"] == "dry_run":
            print(f"🧪 DRY RUN: {result['message']}")
            for table, indexes in result["missing_indexes"].items():
                print(f"\n📋 {table}:")
                for idx_name, columns in indexes:
                    print(f"   - {idx_name}: ({', '.join(columns)})")
        else:
            print(f"📊 Index Creation Results:")
            print(f"   Created: {result['total_created']}")
            print(f"   Errors: {result['total_errors']}")

            if result["created"]:
                print(f"\n✅ Successfully created indexes:")
                for idx in result["created"]:
                    print(
                        f"   - {idx['name']} on {idx['table']} ({idx['creation_time_seconds']:.2f}s)"
                    )

    elif args.command == "optimize":
        vacuum = args.vacuum
        analyze = args.analyze

        if not vacuum and not analyze:
            # Default to both if none specified
            vacuum = analyze = True

        result = optimizer.optimize_database(vacuum=vacuum, analyze=analyze)

        print(f"📊 Optimization Results:")
        print(f"   Time taken: {result['time_taken_seconds']:.2f}s")
        print(f"   Size before: {result['before_size_mb']:.2f} MB")
        print(f"   Size after: {result['after_size_mb']:.2f} MB")
        if result["space_saved_mb"] > 0:
            print(f"   Space saved: {result['space_saved_mb']:.2f} MB")

        for op in result["operations"]:
            print(
                f"   ✅ {op['operation']}: {op['description']} ({op['time_seconds']:.2f}s)"
            )

    elif args.command == "full-optimize":
        print("🚀 Running full database optimization...")

        # Create missing indexes
        print("\n1️⃣ Creating missing indexes...")
        index_result = optimizer.create_missing_indexes()
        if index_result["total_created"] > 0:
            print(f"   ✅ Created {index_result['total_created']} indexes")

        # Optimize database
        print("\n2️⃣ Optimizing database...")
        opt_result = optimizer.optimize_database(vacuum=True, analyze=True)
        print(
            f"   ✅ Optimization completed in {opt_result['time_taken_seconds']:.2f}s"
        )
        if opt_result["space_saved_mb"] > 0:
            print(f"   💾 Space saved: {opt_result['space_saved_mb']:.2f} MB")

        print("\n🎉 Full optimization complete!")

    elif args.command == "integrity":
        result = optimizer.check_data_integrity()

        print(f"🔒 DATA INTEGRITY CHECK")
        print("=" * 40)
        print(f"Status: {result['status'].upper()}")
        print(f"Total issues: {result['total_issues']}")

        if result["total_issues"] > 0:
            for issue_type in [
                "foreign_key_violations",
                "data_consistency_issues",
                "missing_data_issues",
                "duplicate_issues",
            ]:
                issues = result[issue_type]
                if issues:
                    print(f"\n{issue_type.replace('_', ' ').title()}:")
                    for issue in issues:
                        print(f"   - {issue}")
        else:
            print("\n✅ No integrity issues found")

    elif args.command == "stats":
        stats = optimizer.get_database_statistics()

        print(f"📊 DATABASE STATISTICS")
        print("=" * 40)
        print(f"File: {stats['database_info']['file_path']}")
        print(f"Size: {stats['database_info']['database_size_mb']:.2f} MB")
        print(f"Pages: {stats['database_info']['page_count']:,}")

        print(f"\nTables:")
        for table, table_stats in stats["tables"].items():
            print(f"   {table}: {table_stats['row_count']:,} rows")


if __name__ == "__main__":
    main()
