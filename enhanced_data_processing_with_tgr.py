#!/usr/bin/env python3
"""
Enhanced Data Processing Integration with TGR Scraper
====================================================

This module integrates the comprehensive TGR scraper functionality
into the existing enhanced data processing workflow.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging

# Add src to path
sys.path.insert(0, 'src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def enhanced_data_processing_with_tgr(safe_log_func=None, progress_callback=None):
    """
    Enhanced data processing that includes comprehensive TGR data collection.
    
    Args:
        safe_log_func: Function to log messages to UI (optional)
        progress_callback: Function to update progress (optional)
    
    Returns:
        dict: Results summary including TGR data statistics
    """
    
    def log_message(msg, level="INFO", progress=None):
        """Helper to log messages"""
        logger.info(msg)
        if safe_log_func:
            safe_log_func(msg, level, progress)
        if progress_callback and progress is not None:
            progress_callback(progress)
    
    results = {
        'status': 'success',
        'traditional_processing': {},
        'tgr_processing': {},
        'total_dogs_processed': 0,
        'total_races_processed': 0,
        'files_generated': [],
        'processing_time': 0,
        'errors': []
    }
    
    start_time = datetime.now()
    
    try:
        log_message("🚀 Starting Enhanced Data Processing with TGR Integration...", "INFO", 0)
        
        # Phase 1: Traditional Data Processing (existing files)
        log_message("📊 Phase 1: Processing traditional CSV form guides...", "INFO", 10)
        
        try:
            # Try to use existing enhanced processor for traditional data
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "enhanced_comprehensive_processor",
                "./enhanced_comprehensive_processor.py"
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                EnhancedComprehensiveProcessor = module.EnhancedComprehensiveProcessor
                
                processor = EnhancedComprehensiveProcessor()
                traditional_results = processor.process_all_unprocessed()
                results['traditional_processing'] = traditional_results
                
                log_message(f"✅ Traditional processing: {traditional_results.get('processed_count', 0)} files", "INFO", 30)
            else:
                log_message("⚠️ Enhanced processor not available, skipping traditional processing", "WARNING", 30)
                results['traditional_processing'] = {'status': 'skipped', 'message': 'Processor not available'}
                
        except Exception as e:
            log_message(f"⚠️ Traditional processing failed: {e}", "WARNING", 30)
            results['traditional_processing'] = {'status': 'failed', 'error': str(e)}
        
        # Phase 2: TGR Enhanced Data Collection
        log_message("🐕 Phase 2: Comprehensive TGR data collection...", "INFO", 35)
        
        try:
            from collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
            
            # Initialize TGR scraper
            log_message("🔧 Initializing TGR scraper...", "INFO", 40)
            scraper = TheGreyhoundRecorderScraper(rate_limit=2.0, use_cache=True)
            
            # Discover all cached race files and extract comprehensive data
            log_message("🔍 Discovering cached TGR race files...", "INFO", 45)
            
            cache_dir = Path('.tgr_cache')
            if cache_dir.exists():
                cached_files = list(cache_dir.glob('*.html'))
                log_message(f"📁 Found {len(cached_files)} cached TGR files", "INFO", 50)
                
                if cached_files:
                    # Run comprehensive TGR data extraction
                    log_message("📊 Running comprehensive TGR data extraction...", "INFO", 55)
                    
                    # Import our comprehensive collection function
                    sys.path.insert(0, os.getcwd())
                    from run_full_enhanced_collection import run_full_enhanced_collection
                    
                    # Run the full enhanced collection
                    tgr_success = run_full_enhanced_collection()
                    
                    if tgr_success:
                        log_message("✅ TGR comprehensive data collection completed!", "INFO", 75)
                        
                        # Check for generated files
                        timestamp_pattern = datetime.now().strftime("%Y%m%d")
                        tgr_files = []
                        
                        for pattern in [f"enhanced_race_data_{timestamp_pattern}*.json", 
                                      f"enhanced_dog_data_{timestamp_pattern}*.json",
                                      f"enhanced_analytics_{timestamp_pattern}*.json"]:
                            matches = list(Path('.').glob(pattern))
                            tgr_files.extend(matches)
                        
                        if tgr_files:
                            log_message(f"📁 Generated {len(tgr_files)} TGR data files", "INFO", 80)
                            results['files_generated'].extend([str(f) for f in tgr_files])
                            
                            # Try to get statistics from analytics file
                            analytics_files = [f for f in tgr_files if 'analytics' in str(f)]
                            if analytics_files:
                                try:
                                    with open(analytics_files[0], 'r') as f:
                                        analytics = json.load(f)
                                    
                                    summary = analytics.get('summary', {})
                                    results['total_dogs_processed'] = summary.get('total_unique_dogs', 0)
                                    results['total_races_processed'] = summary.get('total_history_records', 0)
                                    
                                    log_message(f"📊 TGR Statistics: {results['total_dogs_processed']} dogs, {results['total_races_processed']} race records", "INFO", 85)
                                    
                                except Exception as e:
                                    log_message(f"⚠️ Could not read analytics: {e}", "WARNING")
                        
                        results['tgr_processing'] = {
                            'status': 'success',
                            'files_processed': len(cached_files),
                            'files_generated': len(tgr_files),
                            'dogs_processed': results['total_dogs_processed'],
                            'races_processed': results['total_races_processed']
                        }
                    else:
                        log_message("❌ TGR comprehensive data collection failed", "ERROR", 60)
                        results['tgr_processing'] = {'status': 'failed', 'error': 'Collection process failed'}
                        results['errors'].append("TGR data collection failed")
                else:
                    log_message("ℹ️ No cached TGR files found", "INFO", 60)
                    results['tgr_processing'] = {'status': 'skipped', 'message': 'No cached data available'}
            else:
                log_message("ℹ️ TGR cache directory not found", "INFO", 60)
                results['tgr_processing'] = {'status': 'skipped', 'message': 'Cache directory not found'}
                
        except ImportError as e:
            log_message(f"❌ TGR scraper not available: {e}", "ERROR", 60)
            results['tgr_processing'] = {'status': 'failed', 'error': f'TGR scraper import failed: {e}'}
            results['errors'].append(f"TGR scraper unavailable: {e}")
        except Exception as e:
            log_message(f"❌ TGR processing failed: {e}", "ERROR", 60)
            results['tgr_processing'] = {'status': 'failed', 'error': str(e)}
            results['errors'].append(f"TGR processing error: {e}")
        
        # Phase 3: Data Integration & Summary
        log_message("🔄 Phase 3: Finalizing enhanced data processing...", "INFO", 90)
        
        # Calculate totals
        traditional_count = results.get('traditional_processing', {}).get('processed_count', 0)
        tgr_dogs = results.get('total_dogs_processed', 0)
        tgr_races = results.get('total_races_processed', 0)
        
        # Final summary
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = processing_time
        
        if results['errors']:
            results['status'] = 'partial_success'
            log_message(f"⚠️ Completed with {len(results['errors'])} errors", "WARNING", 95)
        else:
            log_message("✅ All processing phases completed successfully!", "INFO", 95)
        
        # Comprehensive summary
        summary_msg = f"🎉 Enhanced Data Processing Complete!"
        if traditional_count > 0:
            summary_msg += f"\n   • Traditional files: {traditional_count} processed"
        if tgr_dogs > 0:
            summary_msg += f"\n   • TGR data: {tgr_dogs} dogs, {tgr_races} race records"
        if results['files_generated']:
            summary_msg += f"\n   • Files generated: {len(results['files_generated'])}"
        summary_msg += f"\n   • Processing time: {processing_time:.1f}s"
        
        log_message(summary_msg, "INFO", 100)
        
    except Exception as e:
        log_message(f"❌ Critical error in enhanced data processing: {e}", "ERROR", 100)
        results['status'] = 'failed'
        results['errors'].append(f"Critical error: {e}")
    
    return results


def integrate_tgr_into_app_processing():
    """
    Integration function that can be called from app.py to include TGR processing
    in the enhanced data processing workflow.
    """
    
    # This would replace the process_data_background function in app.py
    def enhanced_process_data_background():
        """Enhanced background task that includes TGR data processing"""
        global processing_status
        
        # Import processing status management from app.py context
        # This would need to be properly integrated
        
        try:
            # Use our enhanced processing function
            def safe_log_wrapper(msg, level, progress=None):
                # This would call the safe_log_to_processing function from app.py
                print(f"[{level}] {msg}")
                if progress is not None:
                    print(f"Progress: {progress}%")
            
            def progress_wrapper(progress):
                # This would update the processing_status progress
                print(f"Progress: {progress}%")
            
            # Run enhanced processing with TGR integration
            results = enhanced_data_processing_with_tgr(
                safe_log_func=safe_log_wrapper,
                progress_callback=progress_wrapper
            )
            
            return results
            
        except Exception as e:
            print(f"Error in enhanced processing: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    return enhanced_process_data_background


if __name__ == "__main__":
    # Test the enhanced processing
    print("🧪 Testing Enhanced Data Processing with TGR Integration")
    results = enhanced_data_processing_with_tgr()
    
    print("\n📊 Results Summary:")
    print(f"Status: {results['status']}")
    print(f"Dogs processed: {results['total_dogs_processed']}")
    print(f"Races processed: {results['total_races_processed']}")
    print(f"Files generated: {len(results['files_generated'])}")
    print(f"Processing time: {results['processing_time']:.1f}s")
    
    if results['errors']:
        print(f"\n⚠️ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
