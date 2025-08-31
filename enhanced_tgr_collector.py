#!/usr/bin/env python3
"""
Enhanced TGR Data Collector
===========================

Comprehensive data collection from The Greyhound Recorder
with advanced parsing and storage capabilities.
"""

import logging
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our enhanced components
try:
    from src.collectors.the_greyhound_recorder_scraper import TheGreyhoundRecorderScraper
    from tgr_prediction_integration import TGRPredictionIntegrator
    HAS_TGR_COMPONENTS = True
except ImportError as e:
    logging.warning(f"TGR components not fully available: {e}")
    HAS_TGR_COMPONENTS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTGRCollector:
    """Enhanced TGR data collector with comprehensive analysis."""
    
    def __init__(self, db_path: str = "greyhound_racing_data.db"):
        self.db_path = db_path
        self.scraper = None
        self.integrator = None
        
        # Initialize components if available
        if HAS_TGR_COMPONENTS:
            try:
                self.scraper = TheGreyhoundRecorderScraper(
                    rate_limit=3.0,  # Be respectful to TGR servers
                    cache_dir=".tgr_cache",
                    use_cache=True
                )
                self.integrator = TGRPredictionIntegrator(db_path=self.db_path)
                logger.info("✅ Enhanced TGR collector initialized with all components")
            except Exception as e:
                logger.error(f"Failed to initialize TGR components: {e}")
        else:
            logger.warning("⚠️ TGR collector running in limited mode")
        
        # Set up database connection
        self.ensure_database_setup()
    
    def ensure_database_setup(self):
        """Ensure the enhanced TGR tables exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if enhanced tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='tgr_dog_performance_summary'
            """)
            
            if not cursor.fetchone():
                logger.warning("Enhanced TGR tables not found. Please run migrate_tgr_enhancement.py first")
            
            conn.close()
        except Exception as e:
            logger.error(f"Database setup check failed: {e}")
    
    def collect_comprehensive_dog_data(self, dog_names: List[str]) -> Dict[str, Any]:
        """Collect comprehensive data for a list of dogs."""
        
        logger.info(f"🔍 Collecting comprehensive TGR data for {len(dog_names)} dogs")
        
        results = {
            'dogs_processed': 0,
            'total_entries': 0,
            'total_insights': 0,
            'dogs_data': {},
            'errors': []
        }
        
        if not self.scraper:
            logger.error("❌ TGR scraper not available")
            results['errors'].append("TGR scraper not initialized")
            return results
        
        for dog_name in dog_names:
            try:
                logger.info(f"Processing dog: {dog_name}")
                start_time = time.time()
                
                # Collect enhanced data from TGR
                enhanced_data = self.scraper.fetch_enhanced_dog_data(dog_name)
                
                if enhanced_data and enhanced_data.get('form_entries'):
                    # Store to database
                    self._store_enhanced_dog_data(enhanced_data)
                    
                    # Calculate and store performance metrics
                    self._update_performance_summary(enhanced_data)
                    
                    # Store expert insights
                    self._store_expert_insights(enhanced_data)
                    
                    # Log the collection
                    self._log_collection_activity(
                        dog_name, 
                        'enhanced_data',
                        'success',
                        len(enhanced_data.get('form_entries', [])),
                        len(enhanced_data.get('recent_comments', [])),
                        time.time() - start_time
                    )
                    
                    results['dogs_data'][dog_name] = enhanced_data
                    results['total_entries'] += len(enhanced_data.get('form_entries', []))
                    results['total_insights'] += len(enhanced_data.get('recent_comments', []))
                    
                    logger.info(f"✅ {dog_name}: {len(enhanced_data.get('form_entries', []))} entries, "
                              f"{len(enhanced_data.get('recent_comments', []))} insights")
                else:
                    logger.warning(f"⚠️ No data found for {dog_name}")
                    self._log_collection_activity(
                        dog_name, 'enhanced_data', 'no_data', 0, 0, time.time() - start_time
                    )
                
                results['dogs_processed'] += 1
                
                # Rate limiting
                time.sleep(3.0)  # Be respectful to TGR servers
                
            except Exception as e:
                error_msg = f"Failed to collect data for {dog_name}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                
                self._log_collection_activity(
                    dog_name, 'enhanced_data', 'failed', 0, 0, 0, str(e)
                )
        
        logger.info(f"🏁 Collection complete: {results['dogs_processed']} dogs, "
                   f"{results['total_entries']} entries, {results['total_insights']} insights")
        
        return results
    
    def collect_data_for_recent_races(self, days_back: int = 7) -> Dict[str, Any]:
        """Collect TGR data for dogs in recent races."""
        
        logger.info(f"📅 Collecting TGR data for dogs from races in last {days_back} days")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get dogs from recent races
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
                SELECT DISTINCT d.dog_clean_name
                FROM dog_race_data d
                JOIN race_metadata r ON d.race_id = r.race_id
                WHERE r.race_date >= ?
                  AND d.dog_clean_name IS NOT NULL
                  AND d.dog_clean_name != ''
                ORDER BY r.race_date DESC
                LIMIT 50
            """
            
            cursor.execute(query, [cutoff_date])
            recent_dogs = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            if not recent_dogs:
                logger.warning("No recent dogs found")
                return {'dogs_processed': 0, 'total_entries': 0}
            
            logger.info(f"Found {len(recent_dogs)} dogs from recent races")
            
            # Collect data for these dogs
            return self.collect_comprehensive_dog_data(recent_dogs)
            
        except Exception as e:
            logger.error(f"Failed to collect data for recent races: {e}")
            return {'error': str(e)}
    
    def _store_enhanced_dog_data(self, enhanced_data: Dict[str, Any]):
        """Store enhanced dog form data."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            dog_name = enhanced_data['dog_name']
            
            for entry in enhanced_data.get('form_entries', []):
                # Store in enhanced form table
                cursor.execute("""
                    INSERT OR REPLACE INTO tgr_enhanced_dog_form
                    (dog_name, race_date, venue, grade, distance, box_number,
                     recent_form, weight, comments, odds, odds_text, trainer,
                     profile_url, race_url, field_size, race_number, expert_comments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    dog_name,
                    entry.get('race_date'),
                    entry.get('venue'),
                    entry.get('grade'),
                    entry.get('distance'),
                    entry.get('box_number'),
                    json.dumps(entry.get('recent_form', [])),
                    entry.get('weight'),
                    entry.get('comments'),
                    entry.get('odds'),
                    entry.get('odds_text'),
                    entry.get('trainer'),
                    entry.get('profile_url'),
                    entry.get('race_url'),
                    entry.get('field_size'),
                    entry.get('race_number'),
                    json.dumps(entry.get('expert_comments', []))
                ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store enhanced dog data: {e}")
    
    def _update_performance_summary(self, enhanced_data: Dict[str, Any]):
        """Update performance summary for the dog."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            dog_name = enhanced_data['dog_name']
            performance = enhanced_data.get('performance_summary', {})
            venue_analysis = enhanced_data.get('venue_analysis', {})
            distance_analysis = enhanced_data.get('distance_analysis', {})
            
            cursor.execute("""
                INSERT OR REPLACE INTO tgr_dog_performance_summary
                (dog_name, performance_data, venue_analysis, distance_analysis,
                 total_entries, wins, places, win_percentage, place_percentage,
                 average_position, best_position, consistency_score, form_trend,
                 distance_versatility, venues_raced, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                dog_name,
                json.dumps(performance),
                json.dumps(venue_analysis),
                json.dumps(distance_analysis),
                performance.get('total_starts', 0),
                performance.get('wins', 0),
                performance.get('places', 0),
                performance.get('win_percentage', 0.0),
                performance.get('place_percentage', 0.0),
                performance.get('average_position', 0.0),
                performance.get('best_position', 8),
                performance.get('consistency_score', 0.0),
                performance.get('recent_form_trend', 'stable'),
                performance.get('distance_versatility', 0),
                len(venue_analysis),
                datetime.now().isoformat()
            ])
            
            # Store venue-specific performance
            for venue, stats in venue_analysis.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO tgr_venue_performance
                    (dog_name, venue, starts, wins, places, positions,
                     win_rate, place_rate, average_position, best_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    dog_name, venue,
                    stats.get('starts', 0),
                    stats.get('wins', 0),
                    stats.get('places', 0),
                    json.dumps(stats.get('positions', [])),
                    stats.get('win_rate', 0.0),
                    stats.get('place_rate', 0.0),
                    stats.get('average_position', 0.0),
                    stats.get('best_position', 8)
                ])
            
            # Store distance-specific performance
            for distance, stats in distance_analysis.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO tgr_distance_performance
                    (dog_name, distance, starts, wins, places, positions,
                     win_rate, place_rate, average_position, best_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    dog_name, distance,
                    stats.get('starts', 0),
                    stats.get('wins', 0),
                    stats.get('places', 0),
                    json.dumps(stats.get('positions', [])),
                    stats.get('win_rate', 0.0),
                    stats.get('place_rate', 0.0),
                    stats.get('average_position', 0.0),
                    stats.get('best_position', 8)
                ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update performance summary: {e}")
    
    def _store_expert_insights(self, enhanced_data: Dict[str, Any]):
        """Store expert insights and comments."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            dog_name = enhanced_data['dog_name']
            
            for comment in enhanced_data.get('recent_comments', []):
                # Calculate basic sentiment score
                sentiment_score = self._calculate_sentiment_score(comment['text'])
                
                cursor.execute("""
                    INSERT OR IGNORE INTO tgr_expert_insights
                    (dog_name, comment_type, race_date, venue, comment_text,
                     source, sentiment_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    dog_name,
                    comment['type'],
                    comment.get('race_date'),
                    comment.get('venue'),
                    comment['text'],
                    comment['source'],
                    sentiment_score
                ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store expert insights: {e}")
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate basic sentiment score for text."""
        
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            'strong', 'impressive', 'excellent', 'good', 'fast', 'winner', 
            'placed', 'improving', 'promising', 'talented', 'consistent',
            'reliable', 'competitive', 'quality', 'effective'
        ]
        
        negative_keywords = [
            'slow', 'weak', 'poor', 'struggled', 'disappointing', 'injured',
            'declining', 'inconsistent', 'unreliable', 'outclassed', 'beaten',
            'failed', 'worst', 'terrible', 'awful'
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return 0.0
        
        # Return score between -1 and 1
        return (positive_count - negative_count) / total_count
    
    def _log_collection_activity(self, dog_name: str, scrape_type: str, status: str,
                               entries_found: int, comments_found: int, 
                               duration: float, error_message: Optional[str] = None):
        """Log collection activity to database."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tgr_scraping_log
                (dog_name, scrape_type, status, entries_found, comments_found,
                 scrape_duration, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                dog_name, scrape_type, status, entries_found, 
                comments_found, duration, error_message
            ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Failed to log collection activity: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about TGR data collection."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Performance summaries
            cursor.execute("SELECT COUNT(*) FROM tgr_dog_performance_summary")
            stats['dogs_with_performance_data'] = cursor.fetchone()[0]
            
            # Form entries
            cursor.execute("SELECT COUNT(*) FROM tgr_enhanced_dog_form")
            stats['total_form_entries'] = cursor.fetchone()[0]
            
            # Expert insights
            cursor.execute("SELECT COUNT(*) FROM tgr_expert_insights")
            stats['total_expert_insights'] = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            stats['recent_collection_attempts'] = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM tgr_scraping_log 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            result = cursor.fetchone()
            stats['recent_success_rate'] = result[0] if result[0] else 0.0
            
            # Top performing dogs
            cursor.execute("""
                SELECT dog_name, win_percentage, total_entries
                FROM tgr_dog_performance_summary
                WHERE total_entries >= 5
                ORDER BY win_percentage DESC
                LIMIT 5
            """)
            stats['top_performers'] = [
                {'dog': row[0], 'win_rate': row[1], 'starts': row[2]}
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}
    
    def run_maintenance(self):
        """Run maintenance tasks on TGR data."""
        
        logger.info("🧹 Running TGR data maintenance...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old cache entries
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute("""
                DELETE FROM tgr_enhanced_feature_cache 
                WHERE expires_at < ?
            """, [cutoff_date])
            
            deleted_cache = cursor.rowcount
            
            # Clean old scraping logs
            log_cutoff = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute("""
                DELETE FROM tgr_scraping_log
                WHERE created_at < ?
            """, [log_cutoff])
            
            deleted_logs = cursor.rowcount
            
            # Update expired performance summaries
            stale_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM tgr_dog_performance_summary
                WHERE last_updated < ?
            """, [stale_cutoff])
            
            stale_summaries = cursor.fetchone()[0]
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Maintenance complete: {deleted_cache} cache entries cleaned, "
                       f"{deleted_logs} log entries cleaned, {stale_summaries} summaries need refresh")
            
            return {
                'cache_cleaned': deleted_cache,
                'logs_cleaned': deleted_logs,
                'stale_summaries': stale_summaries
            }
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
            return {'error': str(e)}

def main():
    """Main function for testing the enhanced collector."""
    
    collector = EnhancedTGRCollector()
    
    # Get collection stats
    stats = collector.get_collection_stats()
    print("📊 Current TGR Collection Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*50)
    
    # Test collection with recent race dogs
    print("🧪 Testing enhanced TGR data collection...")
    
    results = collector.collect_data_for_recent_races(days_back=3)
    print(f"\n📋 Collection Results:")
    print(f"   Dogs processed: {results.get('dogs_processed', 0)}")
    print(f"   Total entries: {results.get('total_entries', 0)}")
    print(f"   Total insights: {results.get('total_insights', 0)}")
    
    if results.get('errors'):
        print(f"   Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"     - {error}")
    
    # Run maintenance
    print("\n🧹 Running maintenance...")
    maintenance_results = collector.run_maintenance()
    for key, value in maintenance_results.items():
        print(f"   {key}: {value}")
    
    # Final stats
    final_stats = collector.get_collection_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Dogs with data: {final_stats.get('dogs_with_performance_data', 0)}")
    print(f"   Form entries: {final_stats.get('total_form_entries', 0)}")
    print(f"   Expert insights: {final_stats.get('total_expert_insights', 0)}")
    print(f"   Success rate: {final_stats.get('recent_success_rate', 0):.1f}%")

if __name__ == "__main__":
    main()
