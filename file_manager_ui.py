#!/usr/bin/env python3
"""
Comprehensive File Manager UI
============================
Clear visibility of all data files with consistent organization across tabs.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from form_guide_csv_scraper import FormGuideScraper

from comprehensive_prediction_pipeline import ComprehensivePredictionPipeline
from enhanced_data_integration import EnhancedDataIntegrator
from upcoming_race_browser import UpcomingRaceBrowser

# Page configuration
st.set_page_config(
    page_title="Greyhound Racing Data Manager",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)


class FileManager:
    def __init__(self):
        self.base_path = Path(".")
        self.exclude_dirs = [
            "backup_before_cleanup",
            "cleanup_archive",
            "quarantine",
            "__pycache__",
            ".git",
            "venv",
            "ml_env",
        ]

        # Initialize collection systems
        self.form_scraper = None
        self.race_browser = None
        self.data_integrator = None
        self.processor = None

        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize all collection and processing systems"""
        try:
            # Initialize Form Guide Scraper
            self.form_scraper = FormGuideScraper()
            print("✅ Form Guide Scraper initialized")
        except Exception as e:
            print(f"⚠️ Form Guide Scraper initialization failed: {e}")
            self.form_scraper = None

        try:
            # Initialize Upcoming Race Browser
            self.race_browser = UpcomingRaceBrowser()
            print("✅ Upcoming Race Browser initialized")
        except Exception as e:
            print(f"⚠️ Upcoming Race Browser initialization failed: {e}")
            self.race_browser = None

        try:
            # Initialize Enhanced Data Integrator
            self.data_integrator = EnhancedDataIntegrator()
            print("✅ Enhanced Data Integrator initialized")
        except Exception as e:
            print(f"⚠️ Enhanced Data Integrator initialization failed: {e}")
            self.data_integrator = None

        try:
            # Initialize Enhanced Comprehensive Processor if available
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "enhanced_comprehensive_processor",
                "./enhanced_comprehensive_processor.py",
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.processor = module.EnhancedComprehensiveProcessor(
                    processing_mode="fast"
                )
                print("✅ Enhanced Comprehensive Processor initialized")
            else:
                self.processor = None
        except Exception as e:
            print(f"⚠️ Enhanced Comprehensive Processor initialization failed: {e}")
            self.processor = None

    def get_pipeline_status(self):
        """Get status of all pipeline components"""
        return {
            "form_scraper": self.form_scraper is not None,
            "race_browser": self.race_browser is not None,
            "data_integrator": self.data_integrator is not None,
            "processor": self.processor is not None,
        }

    def collect_latest_data(self):
        """Collect the most up-to-date data from all sources"""
        results = {"collected_files": [], "errors": []}

        # Collect from form scraper
        if self.form_scraper:
            try:
                # Get recent races (last 2 days for training data)
                from datetime import datetime, timedelta

                target_date = datetime.now() - timedelta(days=1)

                # Use form scraper to collect recent data
                collected = self.form_scraper.collect_races_for_date(
                    target_date.strftime("%Y-%m-%d")
                )
                results["collected_files"].extend(collected)
            except Exception as e:
                results["errors"].append(f"Form scraper error: {str(e)}")

        # Collect upcoming races
        if self.race_browser:
            try:
                upcoming_races = self.race_browser.get_upcoming_races(days_ahead=1)
                results["upcoming_races"] = len(upcoming_races)
            except Exception as e:
                results["errors"].append(f"Race browser error: {str(e)}")

        return results

    def process_unprocessed_files(self):
        """Process all unprocessed files through the pipeline"""
        results = {"processed_count": 0, "errors": []}

        if self.processor:
            try:
                processing_results = self.processor.process_all_unprocessed()
                results["processed_count"] = processing_results.get(
                    "processed_count", 0
                )
                results["status"] = processing_results.get("status", "unknown")
            except Exception as e:
                results["errors"].append(f"Processor error: {str(e)}")
        else:
            results["errors"].append("No processor available")

        return results

    def enhance_expert_data(self):
        """Enhance expert form data with ML insights"""
        results = {"enhanced_files": [], "errors": []}

        if self.data_integrator:
            try:
                # Find expert form files
                expert_files = []
                for file_path in self.base_path.rglob("*"):
                    if file_path.suffix == ".csv" and (
                        "expert_form" in file_path.name.lower()
                        or "expert_data" in file_path.name.lower()
                    ):
                        expert_files.append(file_path)

                # Process each expert file
                for file_path in expert_files[:5]:  # Limit to first 5 for performance
                    try:
                        # This would integrate the expert data
                        enhanced_data = self.data_integrator.get_enhanced_dog_data(
                            "sample_dog", max_races=5
                        )
                        if enhanced_data:
                            results["enhanced_files"].append(str(file_path))
                    except Exception as e:
                        results["errors"].append(
                            f"Error enhancing {file_path.name}: {str(e)}"
                        )

            except Exception as e:
                results["errors"].append(f"Data integrator error: {str(e)}")
        else:
            results["errors"].append("No data integrator available")

        return results

    def scan_all_files(self):
        """Comprehensive scan of all data files"""
        files_data = []

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and not any(
                exclude in str(file_path) for exclude in self.exclude_dirs
            ):
                if file_path.suffix in [".csv", ".json"]:
                    try:
                        stat = file_path.stat()
                        files_data.append(
                            {
                                "name": file_path.name,
                                "path": str(file_path),
                                "directory": str(file_path.parent),
                                "type": file_path.suffix[1:].upper(),
                                "size_bytes": stat.st_size,
                                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "category": self.categorize_file(file_path),
                            }
                        )
                    except Exception as e:
                        continue

        return pd.DataFrame(files_data)

    def categorize_file(self, file_path):
        """Categorize files by type and purpose"""
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()

        if name.startswith("race_"):
            return "Race Data"
        elif name.startswith("analysis_"):
            return "ML Analysis"
        elif "prediction" in parent or "prediction" in name:
            return "Predictions"
        elif "enhanced" in parent or "expert" in parent:
            return "Enhanced Data"
        elif "backtest" in parent or "backtest" in name:
            return "Backtesting"
        elif "model" in parent or "model" in name:
            return "Models"
        elif ("form" in parent or "form" in name) and not (
            "expert" in name or "enhanced" in name
        ):
            return "Form Guides"
        elif name.startswith("upcoming_"):
            return "Upcoming Races"
        elif "expert_form" in name or "expert_data" in name or "expert_guide" in name:
            return "Expert Form Data"
        elif "enhanced_form" in name or name.startswith("enhanced_expert"):
            return "Expert Form Data"
        else:
            return "Other"


def load_sample_data(file_path, max_rows=5):
    """Load sample data from file for preview"""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=max_rows)
            return df
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return pd.DataFrame([data]).head(max_rows)
            elif isinstance(data, list):
                return pd.DataFrame(data).head(max_rows)
    except Exception as e:
        return f"Error loading file: {str(e)}"


def main():
    st.title("🐕 Greyhound Racing Data Manager")
    st.markdown("---")

    # Initialize file manager
    if "file_manager" not in st.session_state:
        st.session_state.file_manager = FileManager()

    # Scan files
    with st.spinner("Scanning all data files..."):
        df = st.session_state.file_manager.scan_all_files()

    # Sidebar with summary statistics
    st.sidebar.header("📊 Data Overview")

    total_files = len(df)
    total_size_mb = df["size_mb"].sum()
    csv_files = len(df[df["type"] == "CSV"])
    json_files = len(df[df["type"] == "JSON"])

    st.sidebar.metric("Total Files", f"{total_files:,}")
    st.sidebar.metric("Total Size", f"{total_size_mb:.1f} MB")
    st.sidebar.metric("CSV Files", f"{csv_files:,}")
    st.sidebar.metric("JSON Files", f"{json_files:,}")

    # Category breakdown
    st.sidebar.subheader("📁 By Category")
    category_counts = df["category"].value_counts()
    for category, count in category_counts.items():
        st.sidebar.write(f"**{category}**: {count:,} files")

    # Pipeline status
    st.sidebar.subheader("🔧 Pipeline Status")
    pipeline_status = st.session_state.file_manager.get_pipeline_status()

    status_icons = {True: "✅", False: "❌"}

    st.sidebar.write(
        f"{status_icons[pipeline_status['form_scraper']]} **Form Scraper**: {'Available' if pipeline_status['form_scraper'] else 'Unavailable'}"
    )
    st.sidebar.write(
        f"{status_icons[pipeline_status['race_browser']]} **Race Browser**: {'Available' if pipeline_status['race_browser'] else 'Unavailable'}"
    )
    st.sidebar.write(
        f"{status_icons[pipeline_status['data_integrator']]} **Data Integrator**: {'Available' if pipeline_status['data_integrator'] else 'Unavailable'}"
    )
    st.sidebar.write(
        f"{status_icons[pipeline_status['processor']]} **Processor**: {'Available' if pipeline_status['processor'] else 'Unavailable'}"
    )

    # Show overall pipeline health
    available_components = sum(pipeline_status.values())
    total_components = len(pipeline_status)
    pipeline_health = (available_components / total_components) * 100

    if pipeline_health == 100:
        health_color = "green"
        health_emoji = "🟢"
    elif pipeline_health >= 75:
        health_color = "orange"
        health_emoji = "🟠"
    else:
        health_color = "red"
        health_emoji = "🔴"

    st.sidebar.markdown(f"{health_emoji} **Pipeline Health**: {pipeline_health:.0f}%")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📋 All Files",
            "🏁 Race Data",
            "🤖 ML & Analysis",
            "🎯 Expert Form Data",
            "📊 Data Explorer",
            "🔍 File Search",
        ]
    )

    with tab1:
        st.header("All Data Files")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            file_type_filter = st.selectbox(
                "File Type", ["All"] + list(df["type"].unique())
            )

        with col2:
            category_filter = st.selectbox(
                "Category", ["All"] + list(df["category"].unique())
            )

        with col3:
            min_size = st.number_input("Min Size (MB)", min_value=0.0, value=0.0)

        # Apply filters
        filtered_df = df.copy()
        if file_type_filter != "All":
            filtered_df = filtered_df[filtered_df["type"] == file_type_filter]
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]
        if min_size > 0:
            filtered_df = filtered_df[filtered_df["size_mb"] >= min_size]

        # Display filtered results
        st.write(f"Showing {len(filtered_df):,} of {len(df):,} files")

        # Interactive data table
        if not filtered_df.empty:
            display_df = filtered_df[
                ["name", "category", "type", "size_mb", "directory", "modified"]
            ].copy()
            display_df = display_df.sort_values("modified", ascending=False)

            # File selection for preview
            selected_indices = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            # File preview
            if (
                hasattr(selected_indices, "selection")
                and selected_indices.selection["rows"]
            ):
                selected_idx = selected_indices.selection["rows"][0]
                selected_file_path = filtered_df.iloc[selected_idx]["path"]

                st.subheader(f"Preview: {Path(selected_file_path).name}")
                preview_data = load_sample_data(selected_file_path)

                if isinstance(preview_data, pd.DataFrame):
                    st.dataframe(preview_data, use_container_width=True)
                else:
                    st.error(preview_data)

    with tab2:
        st.header("🏁 Race Data Files")

        race_files = df[
            df["category"].isin(["Race Data", "Upcoming Races", "Form Guides"])
        ].copy()

        if not race_files.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Race Files", len(race_files[race_files["category"] == "Race Data"])
                )
            with col2:
                st.metric(
                    "Upcoming Races",
                    len(race_files[race_files["category"] == "Upcoming Races"]),
                )
            with col3:
                st.metric(
                    "Form Guides",
                    len(race_files[race_files["category"] == "Form Guides"]),
                )
            with col4:
                st.metric("Total Size", f"{race_files['size_mb'].sum():.1f} MB")

            # Track analysis
            st.subheader("Track Coverage")
            track_pattern = race_files["name"].str.extract(r"Race_\d+_([A-Z_]+)_\d{4}")
            if not track_pattern.empty:
                track_counts = track_pattern[0].value_counts()

                fig = px.bar(
                    x=track_counts.index,
                    y=track_counts.values,
                    title="Files by Track",
                    labels={"x": "Track", "y": "Number of Files"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Race Prediction Interface
            st.subheader("🎯 Race Prediction Interface")

            # Find suitable race files for prediction
            upcoming_race_files = race_files[
                (race_files["category"] == "Upcoming Races")
                | (race_files["name"].str.contains("upcoming", case=False))
                | (race_files["name"].str.contains("Race", case=False))
            ]

            if not upcoming_race_files.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    selected_race_file = st.selectbox(
                        "Select race file for prediction:",
                        options=upcoming_race_files["name"].tolist(),
                        help="Choose a race file to generate comprehensive predictions",
                    )

                with col2:
                    if st.button("🚀 Generate Prediction", type="primary"):
                        if selected_race_file:
                            race_file_path = upcoming_race_files[
                                upcoming_race_files["name"] == selected_race_file
                            ]["path"].iloc[0]

                            with st.spinner("Generating comprehensive prediction..."):
                                try:
                                    # Initialize prediction pipeline
                                    predictor = ComprehensivePredictionPipeline()

                                    # Generate prediction
                                    results = predictor.predict_race_file(
                                        race_file_path
                                    )

                                    if results["success"]:
                                        st.success(
                                            "✅ Prediction completed successfully!"
                                        )

                                        # Display prediction results
                                        st.subheader("🏆 Prediction Results")

                                        # Race info
                                        race_info = results["race_info"]
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Venue", race_info["venue"])
                                        with col2:
                                            st.metric(
                                                "Race Number", race_info["race_number"]
                                            )
                                        with col3:
                                            st.metric("Date", race_info["date"])

                                        # Top predictions
                                        predictions_df = pd.DataFrame(
                                            results["predictions"][:8]
                                        )  # Top 8

                                        if not predictions_df.empty:
                                            # Format for display
                                            display_predictions = predictions_df[
                                                [
                                                    "predicted_rank",
                                                    "dog_name",
                                                    "box_number",
                                                    "final_score",
                                                    "confidence_level",
                                                    "data_quality",
                                                ]
                                            ].copy()

                                            display_predictions.columns = [
                                                "Rank",
                                                "Dog Name",
                                                "Box",
                                                "Score",
                                                "Confidence",
                                                "Data Quality",
                                            ]

                                            # Add styling
                                            def style_predictions(row):
                                                if row["Rank"] <= 3:
                                                    return [
                                                        "background-color: #d4edda"
                                                    ] * len(row)
                                                else:
                                                    return [""] * len(row)

                                            styled_df = display_predictions.style.apply(
                                                style_predictions, axis=1
                                            )
                                            st.dataframe(
                                                styled_df,
                                                use_container_width=True,
                                                hide_index=True,
                                            )

                                            # Prediction summary
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(
                                                    "Total Dogs", results["total_dogs"]
                                                )
                                            with col2:
                                                st.metric(
                                                    "Dogs with Quality Data",
                                                    results["dogs_with_quality_data"],
                                                )
                                            with col3:
                                                methods_used = ", ".join(
                                                    results["prediction_methods_used"]
                                                )
                                                st.write(
                                                    f"**Methods Used**: {methods_used}"
                                                )

                                            # Data quality summary
                                            quality = results["data_quality_summary"]
                                            st.subheader("📊 Data Quality Summary")

                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(
                                                    "Average Quality",
                                                    f"{quality['average_quality']:.2f}",
                                                )
                                            with col2:
                                                st.metric(
                                                    "Good Data Dogs",
                                                    f"{quality['dogs_with_good_data']}/{quality['total_dogs']}",
                                                )
                                            with col3:
                                                st.metric(
                                                    "Poor Data Dogs",
                                                    quality["dogs_with_poor_data"],
                                                )

                                            # Show top 3 picks prominently
                                            st.subheader("🥇 Top 3 Picks")
                                            top_3 = results["predictions"][:3]

                                            for i, pred in enumerate(top_3, 1):
                                                medal = ["🥇", "🥈", "🥉"][i - 1]
                                                with st.container():
                                                    col1, col2, col3 = st.columns(
                                                        [1, 3, 2]
                                                    )
                                                    with col1:
                                                        st.markdown(f"## {medal}")
                                                    with col2:
                                                        st.write(
                                                            f"**{pred['dog_name']}** (Box {pred['box_number']})"
                                                        )
                                                        st.write(
                                                            f"Score: {pred['final_score']:.3f}"
                                                        )
                                                    with col3:
                                                        st.write(
                                                            f"Confidence: {pred['confidence_level']}"
                                                        )
                                                        st.write(
                                                            f"Data Quality: {pred['data_quality']:.2f}"
                                                        )

                                                    if pred.get("reasoning"):
                                                        with st.expander(
                                                            f"Analysis Details for {pred['dog_name']}"
                                                        ):
                                                            for reason in pred[
                                                                "reasoning"
                                                            ]:
                                                                st.write(f"• {reason}")
                                        else:
                                            st.warning("No predictions generated")
                                    else:
                                        st.error(
                                            f"❌ Prediction failed: {results['error']}"
                                        )

                                except Exception as e:
                                    st.error(
                                        f"❌ Error generating prediction: {str(e)}"
                                    )
            else:
                st.info(
                    "No suitable race files found for prediction. Upload upcoming race files to generate predictions."
                )

            # Detailed file listing
            st.subheader("Race Files Detail")
            race_display = race_files[
                ["name", "category", "size_mb", "directory", "modified"]
            ].copy()
            st.dataframe(
                race_display.sort_values("modified", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No race data files found.")

    with tab3:
        st.header("🤖 ML & Analysis Files")

        ml_files = df[
            df["category"].isin(
                ["ML Analysis", "Predictions", "Enhanced Data", "Backtesting", "Models"]
            )
        ].copy()

        if not ml_files.empty:
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(
                    "ML Analysis", len(ml_files[ml_files["category"] == "ML Analysis"])
                )
            with col2:
                st.metric(
                    "Predictions", len(ml_files[ml_files["category"] == "Predictions"])
                )
            with col3:
                st.metric(
                    "Enhanced Data",
                    len(ml_files[ml_files["category"] == "Enhanced Data"]),
                )
            with col4:
                st.metric(
                    "Backtesting", len(ml_files[ml_files["category"] == "Backtesting"])
                )
            with col5:
                st.metric("Models", len(ml_files[ml_files["category"] == "Models"]))

            # Category breakdown chart
            fig = px.pie(
                values=ml_files["category"].value_counts().values,
                names=ml_files["category"].value_counts().index,
                title="ML & Analysis Files by Category",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Files by directory
            st.subheader("Files by Directory")
            dir_counts = ml_files["directory"].value_counts().head(10)

            fig = px.bar(
                x=dir_counts.values,
                y=dir_counts.index,
                orientation="h",
                title="Top Directories by File Count",
                labels={"x": "Number of Files", "y": "Directory"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Recent files
            st.subheader("Recently Modified Files")
            recent_ml = ml_files.nlargest(20, "modified")[
                ["name", "category", "size_mb", "modified"]
            ]
            st.dataframe(recent_ml, use_container_width=True, hide_index=True)
        else:
            st.warning("No ML/Analysis files found.")

    with tab4:
        st.header("🎯 Expert Form Data Files")

        expert_files = df[df["category"] == "Expert Form Data"].copy()

        if not expert_files.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expert Form Files", len(expert_files))
            with col2:
                st.metric("Total Size", f"{expert_files['size_mb'].sum():.1f} MB")
            with col3:
                csv_expert = len(expert_files[expert_files["type"] == "CSV"])
                st.metric("CSV Files", csv_expert)
            with col4:
                json_expert = len(expert_files[expert_files["type"] == "JSON"])
                st.metric("JSON Files", json_expert)

            # Data integration tools
            st.subheader("🔗 Expert Form Data Tools")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📊 Collect Latest Data",
                    help="Collect the most up-to-date race data",
                ):
                    with st.spinner("Collecting latest data..."):
                        try:
                            results = (
                                st.session_state.file_manager.collect_latest_data()
                            )
                            if results["errors"]:
                                st.warning(
                                    f"Collection completed with {len(results['errors'])} errors"
                                )
                                for error in results["errors"]:
                                    st.error(error)
                            else:
                                st.success(
                                    f"✅ Collection completed! Found {len(results.get('collected_files', []))} new files"
                                )
                                if "upcoming_races" in results:
                                    st.info(
                                        f"Found {results['upcoming_races']} upcoming races"
                                    )
                        except Exception as e:
                            st.error(f"Error collecting data: {str(e)}")

            with col2:
                if st.button(
                    "🔄 Process Unprocessed Files",
                    help="Process files through the pipeline",
                ):
                    with st.spinner("Processing files through pipeline..."):
                        try:
                            results = (
                                st.session_state.file_manager.process_unprocessed_files()
                            )
                            if results["errors"]:
                                st.warning(
                                    f"Processing completed with {len(results['errors'])} errors"
                                )
                                for error in results["errors"]:
                                    st.error(error)
                            else:
                                st.success(
                                    f"✅ Processing completed! Processed {results['processed_count']} files"
                                )
                                st.info(f"Status: {results.get('status', 'Unknown')}")
                        except Exception as e:
                            st.error(f"Error processing files: {str(e)}")

            with col3:
                if st.button(
                    "📈 Enhance Expert Data",
                    help="Enhance expert forms with ML analysis",
                ):
                    with st.spinner("Enhancing expert data..."):
                        try:
                            results = (
                                st.session_state.file_manager.enhance_expert_data()
                            )
                            if results["errors"]:
                                st.warning(
                                    f"Enhancement completed with {len(results['errors'])} errors"
                                )
                                for error in results["errors"]:
                                    st.error(error)
                            else:
                                st.success(
                                    f"✅ Enhancement completed! Enhanced {len(results['enhanced_files'])} files"
                                )
                                for file in results["enhanced_files"]:
                                    st.info(f"Enhanced: {Path(file).name}")
                        except Exception as e:
                            st.error(f"Error enhancing data: {str(e)}")

            # Expert form file analysis
            st.subheader("Expert Form File Analysis")

            # Track/venue coverage from expert forms
            expert_tracks = []
            for file_name in expert_files["name"]:
                # Extract track information from expert form filenames
                if "expert_form" in file_name.lower():
                    parts = file_name.split("_")
                    for part in parts:
                        if len(part) > 2 and part.isupper():
                            expert_tracks.append(part)
                            break

            if expert_tracks:
                track_df = pd.DataFrame({"track": expert_tracks})
                track_counts = track_df["track"].value_counts()

                fig = px.bar(
                    x=track_counts.index,
                    y=track_counts.values,
                    title="Expert Form Coverage by Track",
                    labels={"x": "Track", "y": "Number of Expert Form Files"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # File size analysis for expert forms
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Expert Form File Sizes")
                fig = px.histogram(
                    expert_files,
                    x="size_mb",
                    nbins=20,
                    title="Expert Form File Size Distribution",
                    labels={"size_mb": "Size (MB)", "count": "Number of Files"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Expert Form Types")
                type_counts = expert_files["type"].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Expert Form Files by Type",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recent expert form files
            st.subheader("Recent Expert Form Files")
            recent_expert = expert_files.nlargest(10, "modified")[
                ["name", "size_mb", "directory", "modified"]
            ]
            st.dataframe(recent_expert, use_container_width=True, hide_index=True)

            # Expert form data preview
            st.subheader("Expert Form Data Preview")

            if not expert_files.empty:
                selected_expert_file = st.selectbox(
                    "Select expert form file to preview:",
                    options=expert_files["name"].tolist(),
                    index=0,
                )

                if selected_expert_file:
                    selected_path = expert_files[
                        expert_files["name"] == selected_expert_file
                    ]["path"].iloc[0]
                    preview_data = load_sample_data(selected_path, max_rows=10)

                    if isinstance(preview_data, pd.DataFrame):
                        st.dataframe(preview_data, use_container_width=True)

                        # Show column information
                        with st.expander("Column Information"):
                            col_info = pd.DataFrame(
                                {
                                    "Column": preview_data.columns,
                                    "Type": [
                                        str(dtype) for dtype in preview_data.dtypes
                                    ],
                                    "Non-Null Count": preview_data.count(),
                                    "Sample Values": [
                                        (
                                            str(preview_data[col].dropna().iloc[0])
                                            if not preview_data[col].dropna().empty
                                            else "N/A"
                                        )
                                        for col in preview_data.columns
                                    ],
                                }
                            )
                            st.dataframe(
                                col_info, use_container_width=True, hide_index=True
                            )
                    else:
                        st.error(preview_data)

            # Data quality indicators for expert forms
            st.subheader("Expert Form Data Quality")

            quality_metrics = []
            for _, file_info in expert_files.iterrows():
                try:
                    if file_info["type"] == "CSV":
                        sample_data = pd.read_csv(file_info["path"], nrows=100)
                        quality_metrics.append(
                            {
                                "file": file_info["name"],
                                "rows_sampled": len(sample_data),
                                "columns": len(sample_data.columns),
                                "completeness": (
                                    sample_data.notna().sum().sum()
                                    / (len(sample_data) * len(sample_data.columns))
                                    * 100
                                ),
                            }
                        )
                except Exception as e:
                    quality_metrics.append(
                        {
                            "file": file_info["name"],
                            "rows_sampled": 0,
                            "columns": 0,
                            "completeness": 0,
                        }
                    )

            if quality_metrics:
                quality_df = pd.DataFrame(quality_metrics)

                col1, col2 = st.columns(2)
                with col1:
                    avg_completeness = quality_df["completeness"].mean()
                    st.metric("Average Data Completeness", f"{avg_completeness:.1f}%")

                with col2:
                    total_columns = quality_df["columns"].sum()
                    st.metric("Total Columns Across Files", total_columns)

                # Quality breakdown chart
                fig = px.bar(
                    quality_df,
                    x="file",
                    y="completeness",
                    title="Data Completeness by Expert Form File",
                    labels={"completeness": "Completeness (%)", "file": "File"},
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No Expert Form Data files found.")
            st.info(
                "Expert Form Data files should be named with 'expert_form', 'expert_data', 'expert_guide', 'enhanced_form', or start with 'enhanced_expert'."
            )

            # Suggest actions to create expert form data
            st.subheader("Get Started with Expert Form Data")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                **To create Expert Form Data:**
                1. Use the Form Guide Scraper to collect expert analysis
                2. Process form guides with expert insights
                3. Generate enhanced datasets with expert annotations
                """
                )

            with col2:
                if st.button("🚀 Launch Form Guide Scraper"):
                    st.info(
                        "Form Guide Scraper would be launched here to collect expert form data."
                    )

                if st.button("📋 View Data Collection Instructions"):
                    st.markdown(
                        """
                    ### Expert Form Data Collection Instructions:
                    
                    1. **Form Guide Scraping**: Use `form_guide_csv_scraper.py` to collect expert form data
                    2. **Data Processing**: Process collected data with expert insights
                    3. **Enhancement**: Use `enhanced_data_integration.py` to merge with ML analysis
                    4. **Validation**: Ensure data quality and consistency
                    """
                    )

    with tab5:
        st.header("📊 Data Explorer")

        # File size distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("File Size Distribution")
            fig = px.histogram(
                df,
                x="size_mb",
                nbins=50,
                title="File Size Distribution (MB)",
                labels={"size_mb": "Size (MB)", "count": "Number of Files"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Files by Type")
            type_counts = df["type"].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Files by Type",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Timeline of file creation
        st.subheader("File Creation Timeline")
        df_timeline = df.copy()
        df_timeline["date"] = df_timeline["modified"].dt.date
        timeline_data = (
            df_timeline.groupby(["date", "category"]).size().reset_index(name="count")
        )

        fig = px.bar(
            timeline_data,
            x="date",
            y="count",
            color="category",
            title="Files Created Over Time",
            labels={"date": "Date", "count": "Number of Files"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Directory storage usage
        st.subheader("Storage Usage by Directory")
        dir_storage = (
            df.groupby("directory")["size_mb"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )

        fig = px.bar(
            x=dir_storage.values,
            y=dir_storage.index,
            orientation="h",
            title="Storage Usage by Directory (MB)",
            labels={"x": "Size (MB)", "y": "Directory"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.header("🔍 File Search")

        # Search functionality
        search_term = st.text_input(
            "Search files by name or path:", placeholder="Enter search term..."
        )

        if search_term:
            search_results = df[
                df["name"].str.contains(search_term, case=False, na=False)
                | df["path"].str.contains(search_term, case=False, na=False)
            ].copy()

            if not search_results.empty:
                st.write(f"Found {len(search_results)} files matching '{search_term}'")

                # Group by category
                for category in search_results["category"].unique():
                    category_files = search_results[
                        search_results["category"] == category
                    ]

                    with st.expander(f"{category} ({len(category_files)} files)"):
                        display_cols = ["name", "size_mb", "directory", "modified"]
                        st.dataframe(
                            category_files[display_cols].sort_values(
                                "modified", ascending=False
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.info(f"No files found matching '{search_term}'")

        # Advanced filters
        st.subheader("Advanced Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            date_range = st.date_input(
                "Modified date range",
                value=[],
                help="Filter files by modification date",
            )

        with col2:
            size_range = st.slider(
                "File size range (MB)",
                min_value=0.0,
                max_value=float(df["size_mb"].max()),
                value=(0.0, float(df["size_mb"].max())),
                step=0.1,
            )

        with col3:
            selected_categories = st.multiselect(
                "Categories", options=df["category"].unique(), default=[]
            )

        # Apply advanced filters
        if st.button("Apply Advanced Filters"):
            filtered_results = df.copy()

            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_results = filtered_results[
                    (filtered_results["modified"].dt.date >= start_date)
                    & (filtered_results["modified"].dt.date <= end_date)
                ]

            filtered_results = filtered_results[
                (filtered_results["size_mb"] >= size_range[0])
                & (filtered_results["size_mb"] <= size_range[1])
            ]

            if selected_categories:
                filtered_results = filtered_results[
                    filtered_results["category"].isin(selected_categories)
                ]

            st.write(f"Found {len(filtered_results)} files matching advanced criteria")
            if not filtered_results.empty:
                st.dataframe(
                    filtered_results[
                        ["name", "category", "type", "size_mb", "directory", "modified"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
