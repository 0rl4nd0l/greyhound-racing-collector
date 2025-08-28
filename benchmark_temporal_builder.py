#!/usr/bin/env python3
"""
Performance Benchmark: Optimized vs Original Temporal Feature Builder
====================================================================

This script benchmarks the optimized version against the original
temporal feature builder to measure performance improvements.

Metrics measured:
- Total execution time
- Database query count
- Memory usage
- Cache hit rates
- Feature computation time
"""

import time
import tracemalloc
import logging
import pandas as pd
import sqlite3
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import both versions
from temporal_feature_builder import TemporalFeatureBuilder as OriginalBuilder
from temporal_feature_builder_optimized import OptimizedTemporalFeatureBuilder as OptimizedBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark the performance of temporal feature builders."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.results = {}
    
    def create_test_data(self, num_races: int = 5, dogs_per_race: int = 6) -> List[pd.DataFrame]:
        """Create test race data for benchmarking."""
        logger.info(f"Creating test data: {num_races} races with {dogs_per_race} dogs each")
        
        # Get some real race data from the database for testing
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get a sample of recent races
            query = """
            SELECT DISTINCT d.race_id, COUNT(d.dog_clean_name) as dog_count
            FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.race_date IS NOT NULL
            GROUP BY d.race_id
            HAVING dog_count >= ?
            ORDER BY r.race_date DESC
            LIMIT ?
            """
            
            race_ids = pd.read_sql_query(query, conn, params=[dogs_per_race, num_races])
            
            test_races = []
            for _, row in race_ids.iterrows():
                race_id = row['race_id']
                
                # Get full race data
                race_query = """
                SELECT d.*, r.venue, r.grade, r.distance, r.track_condition, r.weather,
                       r.temperature, r.humidity, r.wind_speed, r.field_size,
                       r.race_date, r.race_time
                FROM dog_race_data d
                LEFT JOIN race_metadata r ON d.race_id = r.race_id
                WHERE d.race_id = ?
                ORDER BY d.box_number
                LIMIT ?
                """
                
                race_data = pd.read_sql_query(race_query, conn, params=[race_id, dogs_per_race])
                
                if len(race_data) >= dogs_per_race:
                    test_races.append(race_data)
                
                if len(test_races) >= num_races:
                    break
            
            conn.close()
            
            if len(test_races) < num_races:
                logger.warning(f"Only found {len(test_races)} suitable races (requested {num_races})")
            
            return test_races
        
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            return []
    
    def benchmark_builder(self, builder, test_races: List[pd.DataFrame], 
                         builder_name: str) -> Dict[str, Any]:
        """Benchmark a specific builder implementation."""
        logger.info(f"Benchmarking {builder_name}...")
        
        # Start memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Start timing
        start_time = time.time()
        
        results = {
            'builder_name': builder_name,
            'races_processed': 0,
            'dogs_processed': 0,
            'features_generated': 0,
            'errors': 0,
            'individual_race_times': [],
            'feature_counts': []
        }
        
        # Process each race
        for i, race_data in enumerate(test_races):
            race_id = f"benchmark_race_{i}_{builder_name.lower()}"
            
            try:
                # Time individual race processing
                race_start = time.time()
                
                features_df = builder.build_features_for_race(race_data, race_id)
                
                race_end = time.time()
                race_time = race_end - race_start
                
                # Record results
                results['races_processed'] += 1
                results['dogs_processed'] += len(race_data)
                results['features_generated'] += len(features_df.columns) * len(features_df)
                results['individual_race_times'].append(race_time)
                results['feature_counts'].append(len(features_df.columns))
                
                logger.debug(f"Race {i+1}: {len(race_data)} dogs, {len(features_df.columns)} features, {race_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing race {i+1} with {builder_name}: {e}")
                results['errors'] += 1
        
        # End timing and memory tracking
        end_time = time.time()
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Calculate summary statistics
        total_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        results.update({
            'total_time_seconds': total_time,
            'memory_used_bytes': memory_used,
            'memory_used_mb': memory_used / (1024 * 1024),
            'avg_time_per_race': np.mean(results['individual_race_times']) if results['individual_race_times'] else 0,
            'avg_time_per_dog': total_time / results['dogs_processed'] if results['dogs_processed'] > 0 else 0,
            'races_per_second': results['races_processed'] / total_time if total_time > 0 else 0,
            'dogs_per_second': results['dogs_processed'] / total_time if total_time > 0 else 0,
            'avg_features_per_dog': np.mean(results['feature_counts']) if results['feature_counts'] else 0
        })
        
        # Get additional stats if available (optimized builder)
        if hasattr(builder, 'get_performance_stats'):
            perf_stats = builder.get_performance_stats()
            results['performance_stats'] = perf_stats
        
        return results
    
    def run_benchmark(self, num_races: int = 5, dogs_per_race: int = 6) -> Dict[str, Dict[str, Any]]:
        """Run the complete benchmark comparing both builders."""
        logger.info("🚀 Starting Performance Benchmark")
        logger.info(f"Test parameters: {num_races} races, {dogs_per_race} dogs per race")
        
        # Create test data
        test_races = self.create_test_data(num_races, dogs_per_race)
        
        if not test_races:
            logger.error("No test data available - cannot run benchmark")
            return {}
        
        logger.info(f"Created {len(test_races)} test races")
        
        # Initialize builders
        original_builder = OriginalBuilder(self.db_path)
        optimized_builder = OptimizedBuilder(self.db_path)
        
        results = {}
        
        # Benchmark original builder
        try:
            results['original'] = self.benchmark_builder(
                original_builder, test_races, "Original"
            )
        except Exception as e:
            logger.error(f"Failed to benchmark original builder: {e}")
            results['original'] = {'error': str(e)}
        
        # Clear any potential shared state
        if hasattr(optimized_builder, 'clear_caches'):
            optimized_builder.clear_caches()
        
        # Benchmark optimized builder
        try:
            results['optimized'] = self.benchmark_builder(
                optimized_builder, test_races, "Optimized"
            )
        except Exception as e:
            logger.error(f"Failed to benchmark optimized builder: {e}")
            results['optimized'] = {'error': str(e)}
        
        self.results = results
        return results
    
    def print_comparison_report(self):
        """Print a detailed comparison report."""
        if not self.results or 'original' not in self.results or 'optimized' not in self.results:
            logger.error("No benchmark results available for comparison")
            return
        
        original = self.results['original']
        optimized = self.results['optimized']
        
        # Check for errors
        if 'error' in original:
            logger.error(f"Original builder failed: {original['error']}")
            return
        if 'error' in optimized:
            logger.error(f"Optimized builder failed: {optimized['error']}")
            return
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        # Basic metrics comparison
        print(f"\n📈 EXECUTION TIME COMPARISON:")
        print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 75)
        
        # Total time
        orig_time = original['total_time_seconds']
        opt_time = optimized['total_time_seconds']
        time_improvement = ((orig_time - opt_time) / orig_time * 100) if orig_time > 0 else 0
        print(f"{'Total Time (seconds)':<30} {orig_time:<15.2f} {opt_time:<15.2f} {time_improvement:<15.1f}%")
        
        # Time per race
        orig_race_time = original['avg_time_per_race']
        opt_race_time = optimized['avg_time_per_race']
        race_improvement = ((orig_race_time - opt_race_time) / orig_race_time * 100) if orig_race_time > 0 else 0
        print(f"{'Time per Race (seconds)':<30} {orig_race_time:<15.2f} {opt_race_time:<15.2f} {race_improvement:<15.1f}%")
        
        # Time per dog
        orig_dog_time = original['avg_time_per_dog']
        opt_dog_time = optimized['avg_time_per_dog']
        dog_improvement = ((orig_dog_time - opt_dog_time) / orig_dog_time * 100) if orig_dog_time > 0 else 0
        print(f"{'Time per Dog (seconds)':<30} {orig_dog_time:<15.2f} {opt_dog_time:<15.2f} {dog_improvement:<15.1f}%")
        
        # Throughput
        print(f"\n⚡ THROUGHPUT COMPARISON:")
        print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 75)
        
        orig_rps = original['races_per_second']
        opt_rps = optimized['races_per_second']
        rps_improvement = ((opt_rps - orig_rps) / orig_rps * 100) if orig_rps > 0 else 0
        print(f"{'Races per Second':<30} {orig_rps:<15.2f} {opt_rps:<15.2f} {rps_improvement:<15.1f}%")
        
        orig_dps = original['dogs_per_second']
        opt_dps = optimized['dogs_per_second']
        dps_improvement = ((opt_dps - orig_dps) / orig_dps * 100) if orig_dps > 0 else 0
        print(f"{'Dogs per Second':<30} {orig_dps:<15.2f} {opt_dps:<15.2f} {dps_improvement:<15.1f}%")
        
        # Memory usage
        print(f"\n💾 MEMORY USAGE COMPARISON:")
        print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Change':<15}")
        print("-" * 75)
        
        orig_mem = original['memory_used_mb']
        opt_mem = optimized['memory_used_mb']
        mem_change = ((opt_mem - orig_mem) / orig_mem * 100) if orig_mem > 0 else 0
        print(f"{'Memory Used (MB)':<30} {orig_mem:<15.2f} {opt_mem:<15.2f} {mem_change:<15.1f}%")
        
        # Quality metrics
        print(f"\n📋 QUALITY METRICS:")
        print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Match':<15}")
        print("-" * 75)
        
        orig_features = original['avg_features_per_dog']
        opt_features = optimized['avg_features_per_dog']
        features_match = "✅ Yes" if abs(orig_features - opt_features) < 1 else "❌ No"
        print(f"{'Avg Features per Dog':<30} {orig_features:<15.1f} {opt_features:<15.1f} {features_match:<15}")
        
        print(f"{'Races Processed':<30} {original['races_processed']:<15} {optimized['races_processed']:<15} {'✅ Match' if original['races_processed'] == optimized['races_processed'] else '❌ Mismatch':<15}")
        print(f"{'Dogs Processed':<30} {original['dogs_processed']:<15} {optimized['dogs_processed']:<15} {'✅ Match' if original['dogs_processed'] == optimized['dogs_processed'] else '❌ Mismatch':<15}")
        print(f"{'Errors':<30} {original['errors']:<15} {optimized['errors']:<15} {'✅ Match' if original['errors'] == optimized['errors'] else '❌ Mismatch':<15}")
        
        # Optimization-specific metrics
        if 'performance_stats' in optimized:
            perf_stats = optimized['performance_stats']
            print(f"\n🔧 OPTIMIZATION METRICS (Optimized Builder Only):")
            print(f"Cache Hit Rate: {perf_stats.get('cache_hit_rate', 0):.1%}")
            print(f"Cache Hits: {perf_stats.get('cache_hits', 0)}")
            print(f"Cache Misses: {perf_stats.get('cache_misses', 0)}")
            print(f"DB Queries: {perf_stats.get('db_queries', 0)}")
            print(f"Timestamp Parses: {perf_stats.get('timestamp_parses', 0)}")
            print(f"Feature Computations: {perf_stats.get('feature_computations', 0)}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        if time_improvement > 0:
            print(f"✅ Optimized version is {time_improvement:.1f}% faster")
        else:
            print(f"❌ Optimized version is {abs(time_improvement):.1f}% slower")
        
        if rps_improvement > 0:
            print(f"✅ Optimized version processes {rps_improvement:.1f}% more races per second")
        else:
            print(f"❌ Optimized version processes {abs(rps_improvement):.1f}% fewer races per second")
        
        speedup_factor = orig_time / opt_time if opt_time > 0 else 1
        print(f"📈 Overall speedup factor: {speedup_factor:.2f}x")
        
        print("=" * 80 + "\n")
    
    def save_results(self, filename: str = None):
        """Save benchmark results to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = {}
                for key, value in self.results.items():
                    if isinstance(value, dict):
                        serializable_value = {}
                        for k, v in value.items():
                            if isinstance(v, (list, np.ndarray)):
                                serializable_value[k] = list(v)
                            else:
                                serializable_value[k] = v
                        serializable_results[key] = serializable_value
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Run the performance benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark temporal feature builders")
    parser.add_argument("--races", type=int, default=5, help="Number of races to test")
    parser.add_argument("--dogs", type=int, default=6, help="Dogs per race")
    parser.add_argument("--db", type=str, default="greyhound_racing_data.db", help="Database path")
    parser.add_argument("--save", type=str, help="Save results to file")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = PerformanceBenchmark(args.db)
    results = benchmark.run_benchmark(args.races, args.dogs)
    
    if results:
        # Print comparison report
        benchmark.print_comparison_report()
        
        # Save results if requested
        if args.save:
            benchmark.save_results(args.save)
    else:
        logger.error("Benchmark failed - no results to display")


if __name__ == "__main__":
    main()
