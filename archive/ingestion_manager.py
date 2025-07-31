#!/usr/bin/env python3
"""
Unified Ingestion Manager
=========================

A comprehensive data ingestion system that consolidates FormGuideCsvScraper, 
UpcomingRaceBrowser, and odds scrapers into a single, robust solution with:

- Exponential back-off retry & checksum verification
- Queue-based download scheduler (Asyncio + aiohttp) for parallel processing
- Normalized filename standardization (race_id YYYYMMDD.csv)
- Provenance logging (URL, timestamp, hash) in ingestion_log table
- Schema enforcement with pandas dtype mapping and bad row rejection

Author: AI Assistant
Date: July 31, 2025
"""

import os
import sys
import asyncio
import aiohttp
import sqlite3
import pandas as pd
import hashlib
import time
import random
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import backoff
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class IngestionStatus(Enum):
    """Status tracking for ingestion operations"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class IngestionTask:
    """Data structure for ingestion tasks"""
    task_id: str
    url: str
    source_type: str  # 'form_guide', 'upcoming_race', 'odds'
    race_id: str
    venue: str
    race_date: str
    race_number: int
    priority: int = 1  # Lower numbers = higher priority
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    last_attempt: datetime = None
    status: IngestionStatus = IngestionStatus.PENDING
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class IngestionManager:
    """
    Unified data ingestion manager with comprehensive error handling,
    retry logic, and data quality enforcement.
    """
    
    def __init__(self, 
                 db_path: str = "./databases/race_data.db",
                 base_dir: str = "./",
                 max_concurrent: int = 5,
                 rate_limit_delay: float = 1.0):
        
        self.db_path = db_path
        self.base_dir = Path(base_dir)
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Task queue and session management
        self.task_queue = asyncio.Queue()
        self.active_tasks = set()
        self.session = None
        
        # Venue mapping (consolidated from all scrapers)
        self.venue_map = self._get_venue_mapping()
        
        # Schema enforcement settings
        self.schema_dtypes = self._get_schema_dtypes()
        
        # Setup database
        self.setup_database()
        
        self.logger.info("🚀 Unified Ingestion Manager initialized")
        self.logger.info(f"📂 Base directory: {self.base_dir}")
        self.logger.info(f"🔗 Database: {self.db_path}")
        self.logger.info(f"⚡ Max concurrent: {max_concurrent}")

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "unprocessed",
            "form_guides/downloaded", 
            "upcoming_races",
            "odds_data",
            "logs",
            "databases"
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - INGESTION - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'ingestion_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup database with ingestion_log table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create ingestion_log table for provenance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                race_id TEXT,
                venue TEXT,
                race_date TEXT,
                race_number INTEGER,
                filename TEXT,
                file_hash TEXT,
                file_size INTEGER,
                download_timestamp DATETIME,
                processing_timestamp DATETIME,
                completion_timestamp DATETIME,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                data_quality_score REAL,
                schema_violations INTEGER DEFAULT 0,
                records_imported INTEGER DEFAULT 0,
                records_rejected INTEGER DEFAULT 0,
                provenance_metadata TEXT,  -- JSON metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_race_id ON ingestion_log(race_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_status ON ingestion_log(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_created_at ON ingestion_log(created_at)")
        
        conn.commit()
        conn.close()
        
        self.logger.info("✅ Database setup completed")

    def _get_venue_mapping(self) -> Dict[str, str]:
        """Consolidated venue mapping from all scrapers"""
        return {
            # Major metropolitan tracks
            'angle-park': 'AP_K',
            'sandown': 'SAN',
            'warrnambool': 'WAR',
            'bendigo': 'BEN',
            'geelong': 'GEE',
            'ballarat': 'BAL',
            'horsham': 'HOR',
            'traralgon': 'TRA',
            'dapto': 'DAPT',
            'wentworth-park': 'WPK',
            'albion-park': 'ALBION',
            'cannington': 'CANN',
            'the-meadows': 'MEA',
            'meadows': 'MEA',
            'healesville': 'HEA',
            'sale': 'SAL',
            'richmond': 'RICH',
            'murray-bridge': 'MURR',
            'gawler': 'GAWL',
            'mount-gambier': 'MOUNT',
            'northam': 'NOR',
            'mandurah': 'MAND',
            'gosford': 'GOSF',
            'hobart': 'HOBT',
            'the-gardens': 'GRDN',
            'darwin': 'DARW',
            
            # Additional venues from odds scrapers
            'ladbrokes-q1-lakeside': 'Q1L',
            'townsville': 'TWN',
            'capalaba': 'CAP',
            'ipswich': 'IPS',
            'rockhampton': 'ROCK',
            'bundaberg': 'BUN',
            'cairns': 'CAI',
            'mackay': 'MAC',
            'toowoomba': 'TOO',
            'gold-coast': 'GC',
            'caloundra': 'CAL'
        }

    def _get_schema_dtypes(self) -> Dict[str, Dict[str, str]]:
        """Define strict pandas dtypes for schema enforcement"""
        return {
            'race_metadata': {
                'race_id': 'string',
                'venue': 'string',
                'race_number': 'Int64',
                'race_date': 'string',
                'race_name': 'string',
                'grade': 'string',
                'distance': 'string',
                'track_condition': 'string',
                'field_size': 'Int64',
                'weather': 'string',
                'temperature': 'float64',
                'humidity': 'float64',
                'wind_speed': 'float64'
            },
            'dog_race_data': {
                'race_id': 'string',
                'dog_name': 'string',
                'dog_clean_name': 'string',
                'box_number': 'Int64',
                'trainer_name': 'string',
                'weight': 'float64',
                'odds_decimal': 'float64',
                'starting_price': 'float64',
                'finish_position': 'Int64',
                'margin': 'string',
                'beaten_margin': 'float64',
                'was_scratched': 'boolean'
            },
            'odds_data': {
                'race_id': 'string',
                'dog_name': 'string',
                'box_number': 'Int64',
                'win_odds': 'float64',
                'place_odds': 'float64',
                'bookmaker': 'string',
                'timestamp': 'string'
            }
        }

    def generate_normalized_filename(self, race_id: str, date_str: str, source_type: str) -> str:
        """Generate normalized filename: race_id_YYYYMMDD_source.csv"""
        try:
            # Parse date string and format as YYYYMMDD
            if '-' in date_str:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # Try other common formats
                date_obj = datetime.strptime(date_str, '%d-%m-%Y')
            
            date_formatted = date_obj.strftime('%Y%m%d')
        except:
            # Fallback to current date if parsing fails
            date_formatted = datetime.now().strftime('%Y%m%d')
            self.logger.warning(f"Could not parse date {date_str}, using current date")
        
        # Clean race_id for filename
        clean_race_id = "".join(c for c in race_id if c.isalnum() or c in '-_').rstrip()
        
        return f"{clean_race_id}_{date_formatted}_{source_type}.csv"

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for checksum verification"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def setup_session(self):
        """Setup aiohttp session with proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=20)
        )

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    def add_task(self, task: IngestionTask):
        """Add a task to the ingestion queue"""
        # Log task to database
        self.log_task_to_db(task)
        
        # Add to queue
        asyncio.create_task(self._add_task_to_queue(task))
        
        self.logger.info(f"📋 Added task: {task.task_id} ({task.source_type})")

    async def _add_task_to_queue(self, task: IngestionTask):
        """Internal method to add task to async queue"""
        await self.task_queue.put(task)

    def log_task_to_db(self, task: IngestionTask):
        """Log task creation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        provenance = {
            'priority': task.priority,
            'max_retries': task.max_retries,
            'created_by': 'ingestion_manager',
            'source_details': {
                'venue_mapped': self.venue_map.get(task.venue, task.venue)
            }
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO ingestion_log 
            (task_id, url, source_type, race_id, venue, race_date, race_number, 
             status, retry_count, provenance_metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.url, task.source_type, task.race_id, 
            task.venue, task.race_date, task.race_number,
            task.status.value, task.retry_count, json.dumps(provenance),
            task.created_at, datetime.now()
        ))
        
        conn.commit()
        conn.close()

    def update_task_status(self, task_id: str, status: IngestionStatus, 
                          error_message: str = None, **kwargs):
        """Update task status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = ['status = ?', 'updated_at = ?']
        update_values = [status.value, datetime.now()]
        
        if error_message:
            update_fields.append('error_message = ?')
            update_values.append(error_message)
        
        # Add any additional fields from kwargs
        for key, value in kwargs.items():
            if key in ['filename', 'file_hash', 'file_size', 'download_timestamp',
                      'processing_timestamp', 'completion_timestamp', 'retry_count',
                      'data_quality_score', 'schema_violations', 'records_imported', 
                      'records_rejected']:
                update_fields.append(f'{key} = ?')
                update_values.append(value)
        
        update_values.append(task_id)
        
        cursor.execute(f"""
            UPDATE ingestion_log 
            SET {', '.join(update_fields)}
            WHERE task_id = ?
        """, update_values)
        
        conn.commit()
        conn.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=4,
        base=2,
        max_value=60
    )
    async def download_with_retry(self, task: IngestionTask) -> Tuple[bool, Optional[bytes], str]:
        """Download content with exponential backoff retry"""
        
        self.update_task_status(task.task_id, IngestionStatus.DOWNLOADING,
                              retry_count=task.retry_count)
        
        try:
            self.logger.info(f"⬇️  Downloading: {task.url} (attempt {task.retry_count + 1})")
            
            async with self.session.get(task.url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Verify content is not empty or error page
                    if len(content) < 100:
                        raise aiohttp.ClientError("Content too small, likely error page")
                    
                    # Basic content validation
                    content_str = content.decode('utf-8', errors='ignore')
                    if 'error' in content_str.lower()[:200] or 'not found' in content_str.lower()[:200]:
                        raise aiohttp.ClientError("Error content detected")
                    
                    self.logger.info(f"✅ Downloaded {len(content)} bytes from {task.url}")
                    return True, content, f"Success: {len(content)} bytes"
                    
                else:
                    error_msg = f"HTTP {response.status}"
                    self.logger.warning(f"❌ Download failed: {error_msg}")
                    raise aiohttp.ClientError(error_msg)
                    
        except Exception as e:
            task.retry_count += 1
            error_msg = f"Download error (attempt {task.retry_count}): {str(e)}"
            self.logger.error(error_msg)
            
            if task.retry_count >= task.max_retries:
                self.update_task_status(task.task_id, IngestionStatus.FAILED, 
                                      error_message=error_msg, retry_count=task.retry_count)
                return False, None, error_msg
            else:
                self.update_task_status(task.task_id, IngestionStatus.RETRYING,
                                      error_message=error_msg, retry_count=task.retry_count)
                raise  # Let backoff handle the retry

    def enforce_schema(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict]:
        """Enforce schema with pandas dtype mapping and reject bad rows"""
        
        if table_name not in self.schema_dtypes:
            self.logger.warning(f"No schema defined for table: {table_name}")
            return df, {'violations': 0, 'rejected_rows': 0, 'errors': []}
        
        schema = self.schema_dtypes[table_name]
        original_count = len(df)
        violations = 0
        rejected_rows = 0
        errors = []
        
        self.logger.info(f"🔍 Enforcing schema for {table_name}: {original_count} rows")
        
        # Apply dtype mapping with error handling
        for column, dtype in schema.items():
            if column in df.columns:
                try:
                    if dtype == 'string':
                        df[column] = df[column].astype('str')
                        df[column] = df[column].replace('nan', None)
                    elif dtype == 'Int64':
                        # Handle nullable integers
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                        # Flag rows where conversion failed
                        null_mask = df[column].isna()
                        if null_mask.any():
                            violations += null_mask.sum()
                            errors.append(f"Column {column}: {null_mask.sum()} non-numeric values converted to null")
                    elif dtype == 'float64':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        null_mask = df[column].isna()
                        if null_mask.any():
                            violations += null_mask.sum()
                    elif dtype == 'boolean':
                        df[column] = df[column].astype('bool', errors='ignore')
                        
                except Exception as e:
                    error_msg = f"Schema enforcement error for column {column}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
        
        # Remove rows with critical data missing (e.g., race_id, dog_name)
        critical_columns = ['race_id'] if 'race_id' in df.columns else []
        if table_name == 'dog_race_data':
            critical_columns.extend(['dog_name'])
        
        for col in critical_columns:
            if col in df.columns:
                before_count = len(df)
                df = df.dropna(subset=[col])
                dropped = before_count - len(df)
                if dropped > 0:
                    rejected_rows += dropped
                    errors.append(f"Rejected {dropped} rows with missing {col}")
        
        final_count = len(df)
        
        results = {
            'violations': violations,
            'rejected_rows': original_count - final_count,
            'errors': errors,
            'original_count': original_count,
            'final_count': final_count
        }
        
        self.logger.info(f"📊 Schema enforcement complete: {final_count}/{original_count} rows passed")
        if violations > 0:
            self.logger.warning(f"⚠️  {violations} schema violations detected")
        if rejected_rows > 0:
            self.logger.warning(f"❌ {rejected_rows} rows rejected due to critical missing data")
        
        return df, results

    async def process_csv_content(self, task: IngestionTask, content: bytes) -> Tuple[bool, Dict]:
        """Process CSV content with schema enforcement"""
        
        self.update_task_status(task.task_id, IngestionStatus.PROCESSING,
                              processing_timestamp=datetime.now())
        
        try:
            # Decode content
            content_str = content.decode('utf-8')
            
            # Generate normalized filename
            filename = self.generate_normalized_filename(
                task.race_id, task.race_date, task.source_type
            )
            
            # Save to appropriate directory
            if task.source_type == 'form_guide':
                save_dir = self.base_dir / "unprocessed"
            elif task.source_type == 'upcoming_race':
                save_dir = self.base_dir / "upcoming_races"
            else:
                save_dir = self.base_dir / "odds_data"
            
            file_path = save_dir / filename
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            
            # Calculate checksum
            file_hash = self.calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            # Load into pandas for schema enforcement
            try:
                df = pd.read_csv(file_path)
                
                # Determine table type based on source
                if task.source_type == 'form_guide':
                    table_name = 'dog_race_data'
                elif task.source_type == 'odds':
                    table_name = 'odds_data'
                else:
                    table_name = 'race_metadata'
                
                # Enforce schema
                cleaned_df, schema_results = self.enforce_schema(df, table_name)
                
                # Save cleaned version if changes were made
                if schema_results['rejected_rows'] > 0 or schema_results['violations'] > 0:
                    cleaned_path = file_path.with_suffix('.cleaned.csv')
                    cleaned_df.to_csv(cleaned_path, index=False)
                    self.logger.info(f"💾 Saved cleaned version: {cleaned_path}")
                
                # Update database with results
                self.update_task_status(
                    task.task_id, IngestionStatus.COMPLETED,
                    filename=filename,
                    file_hash=file_hash,
                    file_size=file_size,
                    completion_timestamp=datetime.now(),
                    data_quality_score=1.0 - (schema_results['violations'] / max(schema_results['original_count'], 1)),
                    schema_violations=schema_results['violations'],
                    records_imported=schema_results['final_count'],
                    records_rejected=schema_results['rejected_rows']
                )
                
                results = {
                    'success': True,
                    'filename': filename,
                    'file_path': str(file_path),
                    'file_hash': file_hash,
                    'file_size': file_size,
                    'records_processed': schema_results['final_count'],
                    'schema_results': schema_results
                }
                
                self.logger.info(f"✅ Processing completed: {filename}")
                return True, results
                
            except Exception as e:
                error_msg = f"CSV processing error: {str(e)}"
                self.logger.error(error_msg)
                
                # Still save the raw file for manual inspection
                self.update_task_status(
                    task.task_id, IngestionStatus.FAILED,
                    error_message=error_msg,
                    filename=filename,
                    file_hash=file_hash,
                    file_size=file_size
                )
                
                return False, {'error': error_msg, 'filename': filename}
                
        except Exception as e:
            error_msg = f"Content processing error: {str(e)}"
            self.logger.error(error_msg)
            self.update_task_status(task.task_id, IngestionStatus.FAILED, error_message=error_msg)
            return False, {'error': error_msg}

    async def process_single_task(self, task: IngestionTask):
        """Process a single ingestion task"""
        
        self.logger.info(f"🔄 Processing task: {task.task_id}")
        
        try:
            # Download content with retry
            success, content, message = await self.download_with_retry(task)
            
            if success and content:
                # Process the content
                process_success, results = await self.process_csv_content(task, content)
                
                if process_success:
                    self.logger.info(f"✅ Task completed successfully: {task.task_id}")
                else:
                    self.logger.error(f"❌ Task processing failed: {task.task_id} - {results.get('error', 'Unknown error')}")
            else:
                self.logger.error(f"❌ Task download failed: {task.task_id} - {message}")
                
        except Exception as e:
            error_msg = f"Unexpected error processing task {task.task_id}: {str(e)}"
            self.logger.error(error_msg)
            self.update_task_status(task.task_id, IngestionStatus.FAILED, error_message=error_msg)
        
        finally:
            # Remove from active tasks
            self.active_tasks.discard(task.task_id)

    async def worker(self):
        """Worker coroutine for processing tasks from the queue"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Add to active tasks
                self.active_tasks.add(task.task_id)
                
                # Process the task
                await self.process_single_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker error: {str(e)}")

    async def run_ingestion(self, timeout: int = 300):
        """Run the ingestion process with multiple workers"""
        
        self.logger.info(f"🚀 Starting ingestion with {self.max_concurrent} workers")
        
        # Setup session
        await self.setup_session()
        
        try:
            # Start workers
            workers = [asyncio.create_task(self.worker()) for _ in range(self.max_concurrent)]
            
            # Wait for all tasks to complete or timeout
            try:
                await asyncio.wait_for(self.task_queue.join(), timeout=timeout)
                self.logger.info("✅ All tasks completed successfully")
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Ingestion timed out after {timeout} seconds")
            
            # Cancel workers
            for worker in workers:
                worker.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
            
        finally:
            # Close session
            await self.close_session()

    def get_ingestion_statistics(self) -> Dict:
        """Get comprehensive ingestion statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                status,
                COUNT(*) as count,
                AVG(data_quality_score) as avg_quality,
                SUM(records_imported) as total_records,
                SUM(schema_violations) as total_violations,
                SUM(records_rejected) as total_rejected
            FROM ingestion_log 
            GROUP BY status
        """)
        
        status_stats = {}
        for row in cursor.fetchall():
            status_stats[row[0]] = {
                'count': row[1],
                'avg_quality_score': row[2] or 0,
                'total_records': row[3] or 0,
                'total_violations': row[4] or 0,
                'total_rejected': row[5] or 0
            }
        
        # Get source type statistics
        cursor.execute("""
            SELECT 
                source_type,
                COUNT(*) as count,
                AVG(data_quality_score) as avg_quality
            FROM ingestion_log 
            GROUP BY source_type
        """)
        
        source_stats = {}
        for row in cursor.fetchall():
            source_stats[row[0]] = {
                'count': row[1],
                'avg_quality_score': row[2] or 0
            }
        
        conn.close()
        
        return {
            'status_breakdown': status_stats,
            'source_breakdown': source_stats,
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        }

    # Integration methods for existing scrapers
    
    def add_form_guide_tasks(self, race_urls: List[Dict]):
        """Add form guide scraping tasks from FormGuideCsvScraper data"""
        for race_info in race_urls:
            task = IngestionTask(
                task_id=f"form_{race_info['race_id']}_{int(time.time())}",
                url=race_info['csv_url'],
                source_type='form_guide',
                race_id=race_info['race_id'],
                venue=race_info.get('venue', 'unknown'),
                race_date=race_info.get('race_date', datetime.now().strftime('%Y-%m-%d')),
                race_number=race_info.get('race_number', 1),
                priority=2  # Medium priority for historical form guides
            )
            self.add_task(task)

    def add_upcoming_race_tasks(self, upcoming_races: List[Dict]):
        """Add upcoming race tasks from UpcomingRaceBrowser data"""
        for race_info in upcoming_races:
            task = IngestionTask(
                task_id=f"upcoming_{race_info['race_id']}_{int(time.time())}",
                url=race_info['url'],
                source_type='upcoming_race',
                race_id=race_info['race_id'],
                venue=race_info.get('venue', 'unknown'),
                race_date=race_info.get('date', datetime.now().strftime('%Y-%m-%d')),
                race_number=race_info.get('race_number', 1),
                priority=1  # High priority for upcoming races
            )
            self.add_task(task)

    def add_odds_tasks(self, odds_urls: List[Dict]):
        """Add odds scraping tasks"""
        for odds_info in odds_urls:
            task = IngestionTask(
                task_id=f"odds_{odds_info['race_id']}_{int(time.time())}",
                url=odds_info['url'],
                source_type='odds',
                race_id=odds_info['race_id'],
                venue=odds_info.get('venue', 'unknown'),
                race_date=odds_info.get('race_date', datetime.now().strftime('%Y-%m-%d')),
                race_number=odds_info.get('race_number', 1),
                priority=1  # High priority for live odds
            )
            self.add_task(task)


async def main():
    """Example usage of the Ingestion Manager"""
    
    # Initialize the manager
    manager = IngestionManager(
        db_path="./databases/race_data.db",
        max_concurrent=3,
        rate_limit_delay=2.0
    )
    
    # Example: Add some test tasks
    test_tasks = [
        {
            'race_id': 'test_race_001',
            'csv_url': 'https://example.com/race1.csv',
            'venue': 'sandown',
            'race_date': '2025-07-31',
            'race_number': 1
        }
    ]
    
    # Add form guide tasks
    manager.add_form_guide_tasks(test_tasks)
    
    # Run ingestion
    await manager.run_ingestion(timeout=600)
    
    # Get statistics
    stats = manager.get_ingestion_statistics()
    print("📊 Ingestion Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    # Run the ingestion manager
    asyncio.run(main())
