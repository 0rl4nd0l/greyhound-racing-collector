#!/usr/bin/env python3
"""
Historical Race Data Updater
============================

This script updates existing race data in your database with enhanced expert form data.
It identifies races that need enrichment and processes them with the enhanced scraper
to add sectional times, performance ratings, and other advanced metrics.

Features:
- Identify races in database that lack enhanced data
- Reconstruct race URLs from existing race metadata
- Process races with enhanced expert form scraper
- Update database with enriched data while preserving existing records
- Handle missing or invalid race URLs gracefully
- Progress tracking and comprehensive reporting
- Batch processing with configurable limits

Author: AI Assistant
Date: July 25, 2025
"""

import os
import sys
import sqlite3
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
from urllib.parse import urljoin

# Import our enhanced components
from enhanced_expert_form_scraper import EnhancedExpertFormScraper
from enhanced_data_processor import EnhancedDataProcessor

class HistoricalRaceDataUpdater:
    def __init__(self):
        self.database_path = "./databases/comprehensive_greyhound_data.db"
        self.update_reports_dir = "./historical_updates"
        self.base_url = "https://www.thedogs.com.au"
        
        # Create directories
        os.makedirs(self.update_reports_dir, exist_ok=True)
        
        # Initialize components
        self.expert_scraper = EnhancedExpertFormScraper()
        self.data_processor = EnhancedDataProcessor()
        
        # Venue mapping for URL reconstruction
        self.venue_url_map = {
            'AP_K': 'angle-park',
            'SAN': 'sandown',
            'WAR': 'warrnambool',
            'BEN': 'bendigo',
            'GEE': 'geelong',
            'BAL': 'ballarat',
            'HOR': 'horsham',
            'TRA': 'traralgon',
            'DAPT': 'dapto',
            'WPK': 'wentworth-park',
            'APWE': 'albion-park',
            'CANN': 'cannington',
            'MEA': 'the-meadows',
            'HEA': 'healesville',
            'SAL': 'sale',
            'RICH': 'richmond',
            'MURR': 'murray-bridge',
            'GAWL': 'gawler',
            'MOUNT': 'mount-gambier',
            'NOR': 'northam',
            'MAND': 'mandurah',
            'CASO': 'casino',
            'GOULBURN': 'goulburn',
            'WARRAGUL': 'warragul',
            'TEMORA': 'temora',
            'GUNNEDAH': 'gunnedah',
            'HOBT': 'hobart',
            'LADBROKES-Q-STRAIGHT': 'ladbrokes-q-straight',
            'GRDN': 'gardens'
        }
        
        # Track update statistics
        self.update_stats = {
            'total_races_found': 0,
            'races_needing_update': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'skipped_races': 0,
            'new_records_created': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
        print("🔄 Historical Race Data Updater initialized")
        print(f"💾 Database: {self.database_path}")
        print(f"📁 Update reports: {self.update_reports_dir}")
    
    def identify_races_needing_update(self, limit: Optional[int] = None, date_range: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """Identify races in the database that need enhanced data updates"""
        print(f"\n🔍 IDENTIFYING RACES NEEDING UPDATE")
        print("=" * 60)
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Build query to find races without enhanced data
            base_query = """
                SELECT DISTINCT rm.race_id, rm.venue, rm.race_number, rm.race_date, rm.race_name, rm.url
                FROM race_metadata rm
                LEFT JOIN enhanced_dog_performance edp ON rm.race_id = edp.race_id
                WHERE edp.race_id IS NULL
            """
            
            # Add date range filter if specified
            if date_range:
                start_date, end_date = date_range
                base_query += f" AND rm.race_date BETWEEN '{start_date}' AND '{end_date}'"
            
            # Add ordering and limit
            base_query += " ORDER BY rm.race_date DESC"
            if limit:
                base_query += f" LIMIT {limit}"
            
            cursor.execute(base_query)
            races = cursor.fetchall()
            
            # Convert to list of dictionaries
            race_list = []
            for race in races:
                race_dict = {
                    'race_id': race[0],
                    'venue': race[1],
                    'race_number': race[2],
                    'race_date': race[3],
                    'race_name': race[4],
                    'stored_url': race[5]
                }
                race_list.append(race_dict)
            
            conn.close()
            
            self.update_stats['total_races_found'] = len(race_list)
            self.update_stats['races_needing_update'] = len(race_list)
            
            print(f"✅ Found {len(race_list)} races needing enhanced data")
            
            if date_range:
                print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
            if limit:
                print(f"🎯 Limited to: {limit} races")
            
            return race_list
            
        except Exception as e:
            print(f"❌ Error identifying races: {e}")
            self.update_stats['errors'].append(f"Race identification error: {str(e)}")
            return []
    
    def reconstruct_race_url(self, race_data: Dict[str, Any]) -> Optional[str]:
        """Reconstruct race URL from race metadata"""
        try:
            # Check if we have a stored URL first
            if race_data.get('stored_url'):
                stored_url = race_data['stored_url']
                if stored_url.startswith('http'):
                    return stored_url
            
            # Reconstruct URL from components
            venue = race_data.get('venue', '')
            race_number = race_data.get('race_number', '')
            race_date = race_data.get('race_date', '')
            race_name = race_data.get('race_name', '')
            
            # Convert venue code to URL format
            venue_url = self.venue_url_map.get(venue)
            if not venue_url:
                print(f"⚠️ Unknown venue code: {venue}")
                return None
            
            # Parse race date
            try:
                if '-' in race_date:
                    # Format: YYYY-MM-DD
                    date_obj = datetime.strptime(race_date, '%Y-%m-%d')
                else:
                    # Format: DD Month YYYY
                    date_obj = datetime.strptime(race_date, '%d %B %Y')
                
                date_str = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                print(f"⚠️ Could not parse race date: {race_date}")
                return None
            
            # Create URL-friendly race name
            if race_name:
                # Convert race name to URL format
                url_race_name = race_name.lower()
                url_race_name = re.sub(r'[^\w\s-]', '', url_race_name)  # Remove special chars
                url_race_name = re.sub(r'\s+', '-', url_race_name)  # Replace spaces with hyphens
                url_race_name = re.sub(r'-+', '-', url_race_name)  # Remove multiple hyphens
                url_race_name = url_race_name.strip('-')  # Remove leading/trailing hyphens
            else:
                url_race_name = f"race-{race_number}"
            
            # Construct the URL
            reconstructed_url = f"{self.base_url}/racing/{venue_url}/{date_str}/{race_number}/{url_race_name}"
            
            return reconstructed_url
            
        except Exception as e:
            print(f"❌ Error reconstructing URL for race {race_data.get('race_id', 'unknown')}: {e}")
            return None
    
    def update_race_with_enhanced_data(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a single race with enhanced expert form data"""
        result = {
            'race_id': race_data.get('race_id'),
            'success': False,
            'url_used': None,
            'records_created': 0,
            'errors': []
        }
        
        try:
            # Reconstruct race URL
            race_url = self.reconstruct_race_url(race_data)
            if not race_url:
                result['errors'].append("Could not reconstruct race URL")
                return result
            
            result['url_used'] = race_url
            
            print(f"🔄 Updating race: {race_data.get('venue')} Race {race_data.get('race_number')} on {race_data.get('race_date')}")
            print(f"🌐 URL: {race_url}")
            
            # Process the race with enhanced scraper
            extraction_success = self.expert_scraper.process_race_url(race_url)
            
            if extraction_success:
                print(f"✅ Enhanced data extraction successful")
                
                # Process the extracted data
                processing_results = self.data_processor.process_comprehensive_json_files()
                
                if processing_results['processed'] > 0:
                    result['records_created'] = processing_results['processed']
                    result['success'] = True
                    print(f"✅ Database updated with {processing_results['processed']} records")
                else:
                    result['errors'].append("No data was processed into database")
                    print(f"⚠️ Enhanced data extracted but not processed into database")
            else:
                result['errors'].append("Enhanced data extraction failed")
                print(f"❌ Enhanced data extraction failed")
            
        except Exception as e:
            error_msg = f"Update error: {str(e)}"
            result['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        return result
    
    def update_historical_races_batch(self, race_list: List[Dict[str, Any]], batch_size: int = 10, delay_range: Tuple[float, float] = (2.0, 5.0)) -> Dict[str, Any]:
        """Update historical races in batches"""
        print(f"\n🏁 UPDATING HISTORICAL RACES IN BATCHES")
        print("=" * 60)
        print(f"📊 Total races to update: {len(race_list)}")
        print(f"📦 Batch size: {batch_size}")
        print(f"⏱️ Delay range: {delay_range[0]}-{delay_range[1]} seconds")
        
        self.update_stats['start_time'] = datetime.now().isoformat()
        
        results = {
            'total_races': len(race_list),
            'successful_updates': 0,
            'failed_updates': 0,
            'skipped_races': 0,
            'total_records_created': 0,
            'update_details': []
        }
        
        # Process races in batches
        for i in range(0, len(race_list), batch_size):
            batch = race_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(race_list) + batch_size - 1) // batch_size
            
            print(f"\n📦 BATCH {batch_num}/{total_batches} ({len(batch)} races)")
            print("-" * 40)
            
            for j, race_data in enumerate(batch):
                race_num_in_batch = j + 1
                overall_race_num = i + j + 1
                
                print(f"\n--- Race {race_num_in_batch}/{len(batch)} (Overall: {overall_race_num}/{len(race_list)}) ---")
                
                # Update the race
                update_result = self.update_race_with_enhanced_data(race_data)
                
                # Record results
                if update_result['success']:
                    results['successful_updates'] += 1
                    results['total_records_created'] += update_result['records_created']
                    self.update_stats['successful_updates'] += 1
                    self.update_stats['new_records_created'] += update_result['records_created']
                    print(f"✅ Update successful")
                else:
                    results['failed_updates'] += 1
                    self.update_stats['failed_updates'] += 1
                    print(f"❌ Update failed: {'; '.join(update_result['errors'])}")
                    self.update_stats['errors'].extend(update_result['errors'])
                
                results['update_details'].append(update_result)
                
                # Add delay between races (except for last race in batch)
                if j < len(batch) - 1:
                    delay = random.uniform(delay_range[0], delay_range[1])
                    print(f"⏸️  Waiting {delay:.1f} seconds...")
                    time.sleep(delay)
            
            # Longer delay between batches (except for last batch)
            if batch_num < total_batches:
                batch_delay = random.uniform(10, 20)
                print(f"\n📦 Batch {batch_num} complete. Waiting {batch_delay:.1f} seconds before next batch...")
                time.sleep(batch_delay)
        
        self.update_stats['end_time'] = datetime.now().isoformat()
        
        # Calculate success rate
        results['success_rate'] = results['successful_updates'] / results['total_races'] * 100 if results['total_races'] > 0 else 0
        
        print(f"\n🎯 BATCH UPDATE COMPLETE")
        print("=" * 60)
        print(f"📊 Results:")
        print(f"   • Total races: {results['total_races']}")
        print(f"   • Successful: {results['successful_updates']} ({results['success_rate']:.1f}%)")
        print(f"   • Failed: {results['failed_updates']}")
        print(f"   • Records created: {results['total_records_created']}")
        
        return results
    
    def update_races_by_date_range(self, start_date: str, end_date: str, max_races: int = 100) -> Dict[str, Any]:
        """Update races within a specific date range"""
        print(f"\n📅 UPDATING RACES BY DATE RANGE")
        print("=" * 60)
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"🎯 Maximum races: {max_races}")
        
        # Identify races in the date range
        race_list = self.identify_races_needing_update(
            limit=max_races,
            date_range=(start_date, end_date)
        )
        
        if not race_list:
            print("⚠️ No races found in date range needing updates")
            return {'success': False, 'message': 'No races found'}
        
        # Update the races
        update_results = self.update_historical_races_batch(race_list)
        
        return update_results
    
    def update_recent_races(self, days_back: int = 30, max_races: int = 50) -> Dict[str, Any]:
        """Update recent races that lack enhanced data"""
        print(f"\n📅 UPDATING RECENT RACES")
        print("=" * 60)
        print(f"📅 Looking back: {days_back} days")
        print(f"🎯 Maximum races: {max_races}")
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        return self.update_races_by_date_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            max_races
        )
    
    def generate_update_report(self) -> Dict[str, Any]:
        """Generate comprehensive update report"""
        print(f"\n📊 GENERATING UPDATE REPORT")
        print("=" * 60)
        
        # Get current database statistics
        db_stats = self.get_current_database_stats()
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'update_type': 'historical_race_enhancement',
            'update_statistics': self.update_stats,
            'database_statistics': db_stats,
            'system_health': self.assess_update_health(),
            'recommendations': self.generate_update_recommendations()
        }
        
        # Save report
        report_filename = f"historical_update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.update_reports_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Update report saved: {report_path}")
        
        # Print summary
        self.print_update_summary(report)
        
        return report
    
    def get_current_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        stats = {}
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Count records in various tables
            tables = {
                'race_metadata': 'Total races in system',
                'enhanced_dog_performance': 'Races with enhanced data',
                'sectional_analysis': 'Races with sectional analysis',
                'track_performance_metrics': 'Races with track metrics'
            }
            
            for table, description in tables.items():
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT race_id) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[table] = {'count': count, 'description': description}
                except sqlite3.OperationalError:
                    stats[table] = {'count': 0, 'description': f"{description} (table not found)"}
            
            # Calculate coverage percentage
            total_races = stats['race_metadata']['count']
            enhanced_races = stats['enhanced_dog_performance']['count']
            
            if total_races > 0:
                coverage_percentage = (enhanced_races / total_races) * 100
                stats['coverage'] = {
                    'percentage': coverage_percentage,
                    'enhanced_races': enhanced_races,
                    'total_races': total_races
                }
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def assess_update_health(self) -> Dict[str, Any]:
        """Assess the health of the update process"""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'score': 100
        }
        
        # Check success rate
        total_attempts = self.update_stats['successful_updates'] + self.update_stats['failed_updates']
        if total_attempts > 0:
            success_rate = (self.update_stats['successful_updates'] / total_attempts) * 100
            
            if success_rate < 70:
                health['issues'].append(f"Low update success rate: {success_rate:.1f}%")
                health['score'] -= 25
            elif success_rate < 85:
                health['warnings'].append(f"Moderate update success rate: {success_rate:.1f}%")
                health['score'] -= 10
        
        # Check error count
        error_count = len(self.update_stats['errors'])
        if error_count > 20:
            health['issues'].append(f"High error count: {error_count}")
            health['score'] -= 20
        elif error_count > 10:
            health['warnings'].append(f"Moderate error count: {error_count}")
            health['score'] -= 10
        
        # Determine overall status
        if health['score'] < 70:
            health['status'] = 'unhealthy'
        elif health['score'] < 85:
            health['status'] = 'warning'
        
        return health
    
    def generate_update_recommendations(self) -> List[str]:
        """Generate recommendations based on update results"""
        recommendations = []
        
        # Based on success rate
        total_attempts = self.update_stats['successful_updates'] + self.update_stats['failed_updates']
        if total_attempts > 0:
            success_rate = (self.update_stats['successful_updates'] / total_attempts) * 100
            
            if success_rate < 80:
                recommendations.append("Low success rate detected - review error logs and adjust URL reconstruction logic")
        
        # Based on errors
        error_count = len(self.update_stats['errors'])
        if error_count > 10:
            recommendations.append("High error count - consider implementing more robust error handling and retry logic")
        
        # General recommendations
        recommendations.extend([
            "Monitor enhanced data coverage and prioritize recent races for updates",
            "Consider running updates during off-peak hours to reduce server load",
            "Implement data validation to ensure enhanced data quality",
            "Regular database maintenance and optimization recommended"
        ])
        
        return recommendations
    
    def print_update_summary(self, report: Dict[str, Any]):
        """Print summary of update report"""
        print(f"\n📋 UPDATE SUMMARY")
        print("=" * 60)
        
        # Update statistics
        stats = report['update_statistics']
        print(f"🔄 Update Statistics:")
        print(f"   • Total races found: {stats['total_races_found']}")
        print(f"   • Races needing update: {stats['races_needing_update']}")
        print(f"   • Successful updates: {stats['successful_updates']}")
        print(f"   • Failed updates: {stats['failed_updates']}")
        print(f"   • New records created: {stats['new_records_created']}")
        print(f"   • Errors: {len(stats['errors'])}")
        
        # Database statistics
        db_stats = report['database_statistics']
        if 'coverage' in db_stats:
            coverage = db_stats['coverage']
            print(f"\n💾 Database Coverage:")
            print(f"   • Total races: {coverage['total_races']}")
            print(f"   • Enhanced races: {coverage['enhanced_races']}")
            print(f"   • Coverage: {coverage['percentage']:.1f}%")
        
        # Health assessment
        health = report['system_health']
        print(f"\n🏥 Update Health: {health['status'].upper()} (Score: {health['score']}/100)")
    
    def run_comprehensive_historical_update(self, max_races: int = 100, days_back: int = 90) -> Dict[str, Any]:
        """Run comprehensive historical update process"""
        print(f"\n🚀 COMPREHENSIVE HISTORICAL UPDATE")
        print("=" * 60)
        print(f"🎯 Maximum races: {max_races}")
        print(f"📅 Days back: {days_back}")
        
        comprehensive_results = {
            'start_time': datetime.now().isoformat(),
            'phases': {}
        }
        
        try:
            # Phase 1: Identify races needing updates
            print(f"\n📍 PHASE 1: Identifying Races")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            race_list = self.identify_races_needing_update(
                limit=max_races,
                date_range=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            )
            
            comprehensive_results['phases']['identification'] = {
                'races_found': len(race_list),
                'success': len(race_list) > 0
            }
            
            if not race_list:
                print("✅ No races need updating - all historical races are enhanced")
                comprehensive_results['success'] = True
                comprehensive_results['message'] = "No updates needed"
                return comprehensive_results
            
            # Phase 2: Update races with enhanced data
            print(f"\n📍 PHASE 2: Updating Races")
            update_results = self.update_historical_races_batch(race_list, batch_size=5)
            comprehensive_results['phases']['updates'] = update_results
            
            # Phase 3: Generate report
            print(f"\n📍 PHASE 3: Generating Report")
            report = self.generate_update_report()
            comprehensive_results['phases']['reporting'] = {
                'report_generated': True,
                'report_path': os.path.join(self.update_reports_dir, f"historical_update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            }
            
            comprehensive_results['success'] = True
            comprehensive_results['end_time'] = datetime.now().isoformat()
            
            print(f"\n✅ COMPREHENSIVE UPDATE COMPLETED")
            
        except Exception as e:
            print(f"\n❌ COMPREHENSIVE UPDATE FAILED: {e}")
            comprehensive_results['success'] = False
            comprehensive_results['error'] = str(e)
            comprehensive_results['end_time'] = datetime.now().isoformat()
        
        return comprehensive_results

def main():
    """Main function to demonstrate historical race updating"""
    updater = HistoricalRaceDataUpdater()
    
    print(f"🔄 HISTORICAL RACE DATA UPDATER")
    print("=" * 60)
    print(f"This tool enriches your existing race data with enhanced expert form metrics.")
    
    # Show current database status
    db_stats = updater.get_current_database_stats()
    if 'coverage' in db_stats:
        coverage = db_stats['coverage']
        print(f"\n📊 Current Database Status:")
        print(f"   • Total races: {coverage['total_races']}")
        print(f"   • Enhanced races: {coverage['enhanced_races']}")
        print(f"   • Coverage: {coverage['percentage']:.1f}%")
    
    # Choose update strategy
    print(f"\n🎯 Available Update Options:")
    print(f"   1. Update recent races (last 30 days, max 50 races)")
    print(f"   2. Update by date range (custom)")
    print(f"   3. Comprehensive update (last 90 days, max 100 races)")
    
    # For demonstration, run recent races update
    print(f"\n🚀 Running recent races update (Option 1)...")
    
    try:
        results = updater.update_recent_races(days_back=30, max_races=20)
        
        if results.get('success', False):
            print(f"✅ Recent races update completed successfully")
        else:
            print(f"⚠️ Recent races update completed with issues")
            
    except Exception as e:
        print(f"❌ Update failed: {e}")
    
    # Uncomment for other options:
    """
    # Option 2: Update by specific date range
    results = updater.update_races_by_date_range('2025-07-01', '2025-07-20', max_races=30)
    
    # Option 3: Comprehensive update
    results = updater.run_comprehensive_historical_update(max_races=100, days_back=90)
    """
    
    return results if 'results' in locals() else None

if __name__ == "__main__":
    main()
