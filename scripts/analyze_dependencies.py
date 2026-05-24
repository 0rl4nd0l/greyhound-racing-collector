#!/usr/bin/env python3
"""
Dependency Analysis Script
==========================

Enumerates all pipeline Python files and analyzes their dependencies
focusing on the main three components.
"""

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path


class DependencyAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.main_components = [
            "app.py",
            "ml_system_v3.py",
            "prediction_pipeline_v3.py",
        ]

    def analyze_file(self, filepath):
        """Analyze a Python file for import statements"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])

            return imports
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return set()

    def find_python_files(self):
        """Find all Python files in the repository"""
        python_files = []

        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden directories, node_modules, venv, site-packages, etc.
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in [
                    "node_modules",
                    "venv",
                    "venv_fresh",
                    "__pycache__",
                    "site-packages",
                    ".git",
                    ".vscode",
                    "dist",
                    "build",
                    "env",
                ]
            ]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        return python_files

    def build_dependency_graph(self):
        """Build dependency graph for all Python files"""
        python_files = self.find_python_files()

        # Create mapping of module names to file paths
        module_to_file = {}
        for filepath in python_files:
            module_name = filepath.stem
            module_to_file[module_name] = filepath

        # Analyze dependencies
        for filepath in python_files:
            relative_path = filepath.relative_to(self.root_path)
            imports = self.analyze_file(filepath)

            # Filter to only local dependencies
            local_imports = set()
            for imp in imports:
                if imp in module_to_file:
                    local_imports.add(imp)

            self.dependencies[str(relative_path)] = local_imports

            # Build reverse dependencies
            for dep in local_imports:
                if dep in module_to_file:
                    dep_path = module_to_file[dep].relative_to(self.root_path)
                    self.reverse_dependencies[str(dep_path)].add(str(relative_path))

    def generate_report(self):
        """Generate a dependency analysis report"""
        # Convert sets to lists for JSON serialization
        all_deps_serializable = {k: list(v) for k, v in self.dependencies.items()}
        rev_deps_serializable = {
            k: list(v) for k, v in self.reverse_dependencies.items()
        }

        report = {
            "timestamp": "2025-08-03",
            "total_python_files": len(self.dependencies),
            "main_components": {},
            "all_dependencies": all_deps_serializable,
            "reverse_dependencies": rev_deps_serializable,
        }

        # Analyze main components
        for component in self.main_components:
            if component in self.dependencies:
                deps = self.dependencies[component]
                rev_deps = self.reverse_dependencies.get(component, set())

                report["main_components"][component] = {
                    "dependencies": list(deps),
                    "dependency_count": len(deps),
                    "dependents": list(rev_deps),
                    "dependent_count": len(rev_deps),
                }

        return report

    def create_dot_graph(self):
        """Create a DOT format graph for visualization"""
        dot_content = ["digraph Dependencies {"]
        dot_content.append("  rankdir=TB;")
        dot_content.append("  node [shape=box];")

        # Add main components with special styling
        for component in self.main_components:
            if component in self.dependencies:
                dot_content.append(
                    f'  "{component}" [style=filled, fillcolor=lightblue];'
                )

        # Add edges
        for file, deps in self.dependencies.items():
            for dep in deps:
                # Find the actual file for this dependency
                dep_file = None
                for candidate_file in self.dependencies.keys():
                    if Path(candidate_file).stem == dep:
                        dep_file = candidate_file
                        break

                if dep_file:
                    dot_content.append(f'  "{file}" -> "{dep_file}";')

        dot_content.append("}")
        return "\n".join(dot_content)


def main():
    analyzer = DependencyAnalyzer()
    analyzer.build_dependency_graph()

    # Generate report
    report = analyzer.generate_report()

    # Save JSON report
    with open("audit/code_dependency_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save DOT graph
    dot_graph = analyzer.create_dot_graph()
    with open("audit/code_dependency_graph.dot", "w") as f:
        f.write(dot_graph)

    # Print summary
    print("🔍 Python Code Dependency Analysis")
    print("=" * 50)
    print(f"Total Python files analyzed: {report['total_python_files']}")
    print(f"Analysis saved to: audit/code_dependency_analysis.json")
    print(f"DOT graph saved to: audit/code_dependency_graph.dot")

    print("\n📋 Main Components Analysis:")
    for component, info in report["main_components"].items():
        print(f"\n  {component}:")
        print(f"    Dependencies: {info['dependency_count']}")
        print(f"    Dependents: {info['dependent_count']}")
        if info["dependencies"]:
            print(
                f"    Imports: {', '.join(info['dependencies'][:5])}{'...' if len(info['dependencies']) > 5 else ''}"
            )


if __name__ == "__main__":
    main()
