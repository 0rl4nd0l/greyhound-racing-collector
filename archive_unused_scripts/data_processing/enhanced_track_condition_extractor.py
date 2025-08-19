#!/usr/bin/env python3
"""
Enhanced Track Condition Extractor
==================================

This module provides improved track condition extraction logic that:
1. Avoids false positives from sponsorship text and race names
2. Looks for legitimate track condition sources
3. Validates extracted conditions against context
4. Implements fallback strategies for different data sources

Key improvements:
- Context-aware extraction (avoids sponsorship text) 
- Multiple source validation
- Confidence scoring
- Smart filtering of race name artifacts
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


class EnhancedTrackConditionExtractor:
    def __init__(self):
        self.valid_conditions = {
            'fast': ['fast', 'firm'],
            'good': ['good', 'good4', 'good3'], 
            'slow': ['slow', 'slow5', 'slow6', 'soft'],
            'heavy': ['heavy', 'heavy8', 'heavy9', 'heavy10'],
            'dead': ['dead', 'dead5']
        }
        
        # Sponsorship patterns to avoid
        self.sponsorship_patterns = [
            r'ladbrokes[- ]fast[- ]withdrawals?',
            r'sportsbet[- ]fast[- ]form',
            r'tab[- ]fast[- ]', 
            r'bet365[- ]fast[- ]',
            r'pointsbet[- ]fast[- ]',
            r'unibet[- ]fast[- ]'
        ]
        
    def extract_track_conditions_enhanced(self, soup: BeautifulSoup, race_url: str = "") -> Dict[str, Any]:
        """
        Enhanced track condition extraction with false positive prevention
        """
        try:
            conditions = {}
            extraction_method = None
            confidence_score = 0
            
            print("   🔍 Enhanced track condition extraction starting...")
            
            # Strategy 1: Look for official track condition elements (highest confidence)
            conditions, confidence_score, extraction_method = self._extract_from_official_elements(soup)
            
            if not conditions.get('condition'):
                # Strategy 2: Look for meeting/race information sections
                conditions, confidence_score, extraction_method = self._extract_from_meeting_info(soup)
            
            if not conditions.get('condition'):
                # Strategy 3: Look for structured data (JSON-LD, microdata)
                conditions, confidence_score, extraction_method = self._extract_from_structured_data(soup)
            
            if not conditions.get('condition'):
                # Strategy 4: Pattern matching with context validation
                conditions, confidence_score, extraction_method = self._extract_with_context_validation(soup, race_url)
            
            if not conditions.get('condition'):
                # Strategy 5: Look for venue-specific patterns
                conditions, confidence_score, extraction_method = self._extract_venue_specific(soup, race_url)
            
            # Add extraction metadata
            if conditions:
                conditions['extraction_method'] = extraction_method
                conditions['confidence_score'] = confidence_score
                conditions['extraction_timestamp'] = datetime.now()
                
                # Final validation
                if self._validate_extracted_condition(conditions, race_url):
                    print(f"   ✅ Enhanced extraction successful: {conditions['condition']} (confidence: {confidence_score}, method: {extraction_method})")
                    return conditions
                else:
                    print(f"   ❌ Extracted condition failed validation: {conditions['condition']}")
                    return None
            
            print("   ⚠️ No track conditions found with enhanced extraction")
            return None
            
        except Exception as e:
            print(f"   ❌ Error in enhanced track condition extraction: {e}")
            return None
    
    def _extract_from_official_elements(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """Extract from official track condition elements (highest confidence)"""
        selectors = [
            '.track-condition-official',
            '.official-track-condition', 
            '.track-status',
            '.racing-conditions .track',
            '.meeting-conditions .track-condition',
            '[data-track-condition]',
            '.track-info .condition',
            '.race-track-condition'
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                condition = self._normalize_condition(text)
                if condition:
                    return {
                        'condition': condition,
                        'raw_text': text
                    }, 95, f"official_element_{selector}"
        
        return {}, 0, None
    
    def _extract_from_meeting_info(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """Extract from meeting information sections"""
        # Look for meeting details tables or sections
        meeting_selectors = [
            '.meeting-details',
            '.race-meeting-info', 
            '.track-details',
            '.meeting-header .details',
            'table.meeting-info',
            '.race-info-panel'
        ]
        
        for selector in meeting_selectors:
            section = soup.select_one(selector)
            if section:
                # Look for track condition within this section
                text = section.get_text()
                condition = self._find_condition_in_text(text, exclude_sponsorship=True)
                if condition:
                    return {
                        'condition': condition,
                        'source_section': selector
                    }, 85, f"meeting_info_{selector}"
        
        return {}, 0, None
    
    def _extract_from_structured_data(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """Extract from structured data (JSON-LD, microdata)"""
        # Look for JSON-LD data
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Look for track condition in structured data
                    condition = self._find_in_json_data(data, ['trackCondition', 'condition', 'surface'])
                    if condition:
                        return {
                            'condition': self._normalize_condition(condition),
                            'source': 'json_ld'
                        }, 90, "structured_data_json"
            except:
                continue
        
        # Look for microdata
        microdata_elements = soup.find_all(attrs={'itemscope': True})
        for elem in microdata_elements:
            track_condition = elem.find(attrs={'itemprop': 'trackCondition'})
            if track_condition:
                condition = self._normalize_condition(track_condition.get_text(strip=True))
                if condition:
                    return {
                        'condition': condition,
                        'source': 'microdata'
                    }, 88, "structured_data_microdata"
        
        return {}, 0, None
    
    def _extract_with_context_validation(self, soup: BeautifulSoup, race_url: str) -> Tuple[Dict, int, str]:
        """Extract using pattern matching with context validation"""
        page_text = soup.get_text()
        page_text = re.sub(r'\s+', ' ', page_text)
        
        # Enhanced patterns that avoid sponsorship context
        condition_patterns = [
            # Official track condition statements
            r'track\s+condition[:\s]+([a-z]+)',
            r'surface[:\s]+([a-z]+)',
            r'track[:\s]+([a-z]+)(?:\s+(?:track|condition|surface))?',
            
            # Meeting/race day conditions
            r'(?:today\'?s?|race\s+day)\s+(?:track\s+)?condition[:\s]+([a-z]+)',
            r'meeting\s+condition[:\s]+([a-z]+)',
            
            # Avoiding sponsorship patterns
            r'(?<!ladbrokes\s)(?<!sportsbet\s)(?<!tab\s)(?<!bet365\s)(fast|good|slow|heavy|dead)(?:\s+track)?(?!\s+(?:withdrawals?|form|bet))',
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE)
            for match in matches:
                condition_text = match.group(1).strip().lower()
                
                # Check if this match is in a sponsorship context
                if self._is_in_sponsorship_context(match, page_text, race_url):
                    continue
                    
                condition = self._normalize_condition(condition_text)
                if condition:
                    return {
                        'condition': condition,
                        'context': page_text[max(0, match.start()-50):match.end()+50],
                        'pattern': pattern
                    }, 70, "context_validated_pattern"
        
        return {}, 0, None
    
    def _extract_venue_specific(self, soup: BeautifulSoup, race_url: str) -> Tuple[Dict, int, str]:
        """Extract using venue-specific patterns"""
        # Extract venue from URL
        venue = self._extract_venue_from_url(race_url)
        if not venue:
            return {}, 0, None
        
        # Venue-specific extraction logic
        if venue in ['richmond', 'sandown', 'healesville']:
            # Victorian tracks might have different patterns
            return self._extract_vic_track_pattern(soup)
        elif venue in ['wentworth-park', 'dapto']:
            # NSW tracks
            return self._extract_nsw_track_pattern(soup)
        elif venue in ['angle-park', 'murray-bridge']:
            # SA tracks
            return self._extract_sa_track_pattern(soup)
        
        return {}, 0, None
    
    def _normalize_condition(self, condition_text: str) -> Optional[str]:
        """Normalize track condition to standard values"""
        if not condition_text:
            return None
            
        condition_lower = condition_text.lower().strip()
        
        for standard, variants in self.valid_conditions.items():
            if condition_lower in variants or condition_lower == standard:
                return standard.title()
        
        return None
    
    def _is_in_sponsorship_context(self, match: re.Match, page_text: str, race_url: str) -> bool:
        """Check if a condition match is within sponsorship text"""
        # Check surrounding context
        start = max(0, match.start() - 100)
        end = min(len(page_text), match.end() + 100)
        context = page_text[start:end].lower()
        
        # Check for sponsorship patterns in context
        for pattern in self.sponsorship_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        
        # Check if the condition appears in the race URL (sponsorship indicator)
        if race_url and match.group(1).lower() in race_url.lower():
            return True
        
        # Check for other sponsorship indicators
        sponsorship_indicators = [
            'ladbrokes', 'sportsbet', 'tab', 'bet365', 'pointsbet', 'unibet',
            'withdrawals', 'bonus bet', 'multi bet', 'odds boost'
        ]
        
        for indicator in sponsorship_indicators:
            if indicator in context:
                return True
        
        return False
    
    def _validate_extracted_condition(self, conditions: Dict, race_url: str) -> bool:
        """Final validation of extracted conditions"""
        condition = conditions.get('condition', '').lower()
        
        # Must be a valid condition
        if not any(condition == standard for standard in self.valid_conditions.keys()):
            return False
        
        # Must not be from URL (sponsorship)
        if race_url and condition in race_url.lower():
            return False
        
        # Must have reasonable confidence
        if conditions.get('confidence_score', 0) < 60:
            return False
        
        return True
    
    def _find_condition_in_text(self, text: str, exclude_sponsorship: bool = True) -> Optional[str]:
        """Find track condition in text with optional sponsorship exclusion"""
        text_lower = text.lower()
        
        for standard, variants in self.valid_conditions.items():
            for variant in variants:
                if variant in text_lower:
                    if exclude_sponsorship:
                        # Check if this is in sponsorship context
                        pattern = rf'\b{re.escape(variant)}\b'
                        matches = re.finditer(pattern, text_lower)
                        for match in matches:
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context = text[start:end].lower()
                            
                            # Skip if sponsorship context detected
                            if any(sponsor in context for sponsor in ['ladbrokes', 'sportsbet', 'withdrawals', 'form']):
                                continue
                            
                            return standard.title()
                    else:
                        return standard.title()
        
        return None
    
    def _find_in_json_data(self, data: Any, keys: List[str]) -> Optional[str]:
        """Recursively find track condition in JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in [k.lower() for k in keys]:
                    if isinstance(value, str):
                        return value
                else:
                    result = self._find_in_json_data(value, keys)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_in_json_data(item, keys)
                if result:
                    return result
        
        return None
    
    def _extract_venue_from_url(self, url: str) -> Optional[str]:
        """Extract venue name from race URL"""
        if not url:
            return None
        
        # Extract venue from URL pattern: /racing/{venue}/
        match = re.search(r'/racing/([^/]+)/', url)
        return match.group(1) if match else None
    
    def _extract_vic_track_pattern(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """Victorian track specific extraction patterns"""
        # VIC tracks might have specific HTML structures
        vic_selectors = [
            '.grv-track-condition',
            '.vic-racing-condition',
            '.track-rating'
        ]
        
        for selector in vic_selectors:
            elem = soup.select_one(selector)
            if elem:
                condition = self._normalize_condition(elem.get_text(strip=True))
                if condition:
                    return {'condition': condition}, 75, f"vic_pattern_{selector}"
        
        return {}, 0, None
    
    def _extract_nsw_track_pattern(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """NSW track specific extraction patterns"""
        nsw_selectors = [
            '.grnsw-condition',
            '.nsw-track-info'
        ]
        
        for selector in nsw_selectors:
            elem = soup.select_one(selector)
            if elem:
                condition = self._normalize_condition(elem.get_text(strip=True))
                if condition:
                    return {'condition': condition}, 75, f"nsw_pattern_{selector}"
        
        return {}, 0, None
    
    def _extract_sa_track_pattern(self, soup: BeautifulSoup) -> Tuple[Dict, int, str]:
        """SA track specific extraction patterns"""
        sa_selectors = [
            '.grsa-condition',
            '.sa-track-details'
        ]
        
        for selector in sa_selectors:
            elem = soup.select_one(selector)
            if elem:
                condition = self._normalize_condition(elem.get_text(strip=True))
                if condition:
                    return {'condition': condition}, 75, f"sa_pattern_{selector}"
        
        return {}, 0, None

# Integration function to replace the existing method
def integrate_enhanced_extraction(processor_instance):
    """
    Integrate the enhanced extraction into the existing processor
    """
    extractor = EnhancedTrackConditionExtractor()
    
    # Replace the existing extract_track_conditions method
    def enhanced_extract_track_conditions(soup):
        race_url = getattr(processor_instance, '_current_race_url', '')
        return extractor.extract_track_conditions_enhanced(soup, race_url)
    
    # Monkey patch the method
    processor_instance.extract_track_conditions = enhanced_extract_track_conditions
    
    return processor_instance
