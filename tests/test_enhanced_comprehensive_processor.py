import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_comprehensive_processor import EnhancedComprehensiveProcessor
from bs4 import BeautifulSoup


@pytest.fixture
def processor():
    return EnhancedComprehensiveProcessor()


def test_speed_rating_extraction(processor):
    # Sample HTML with speed ratings
    html_content = '''
    <html>
        <div class="expert-form">
            <table>
                <tr><th>Dog Name</th><th>Speed</th></tr>
                <tr><td>Fast Eddie</td><td>88.5</td></tr>
                <tr><td>Lucky Joe</td><td>79</td></tr>
            </table>
        </div>
    </html>
    '''

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Call the extraction method
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")

    # Expected data (names are converted to uppercase by clean_dog_name method)
    expected_data = {
        'FAST EDDIE': {'speed_rating': 88.5, 'expert_analysis': True},
        'LUCKY JOE': {'speed_rating': 79.0, 'expert_analysis': True}
    }

    # Assert the extracted data matches expected
    assert extracted_data == expected_data


def test_speed_rating_pattern_extraction(processor):
    """Test speed rating extraction from text patterns"""
    html_content = '''
    <html>
        <body>
            <p>Racing analysis: Fast Eddie: 92 Speed Rating</p>
            <p>Another dog: Lucky Joe: 78 Performance Score</p>
            <p>Speed Rating: 85 for Rocket Dog</p>
            <p>SR: 91 for Thunder Bolt</p>
        </body>
    </html>
    '''
    
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")
    
    # Should extract at least Fast Eddie and Thunder Bolt (valid patterns)
    assert 'FAST EDDIE' in extracted_data
    assert extracted_data['FAST EDDIE']['speed_rating'] == 92.0
    assert 'THUNDER BOLT' in extracted_data
    assert extracted_data['THUNDER BOLT']['speed_rating'] == 91.0


def test_speed_rating_validation(processor):
    """Test that invalid speed ratings are filtered out"""
    html_content = '''
    <html>
        <div class="speed-ratings">
            <table>
                <tr><th>Dog</th><th>Rating</th></tr>
                <tr><td>Valid Dog</td><td>75</td></tr>
                <tr><td>Invalid High</td><td>150</td></tr>
                <tr><td>Invalid Low</td><td>-5</td></tr>
                <tr><td>Invalid Text</td><td>abc</td></tr>
                <tr><td>Edge Valid</td><td>100</td></tr>
            </table>
        </div>
    </html>
    '''
    
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")
    
    # Should only extract valid ratings (0-100)
    assert 'VALID DOG' in extracted_data
    assert extracted_data['VALID DOG']['speed_rating'] == 75.0
    assert 'EDGE VALID' in extracted_data
    assert extracted_data['EDGE VALID']['speed_rating'] == 100.0
    
    # Should not extract invalid ratings
    assert 'INVALID HIGH' not in extracted_data
    assert 'INVALID LOW' not in extracted_data
    assert 'INVALID TEXT' not in extracted_data


def test_dog_name_cleaning(processor):
    """Test that dog names are properly cleaned and formatted"""
    html_content = '''
    <html>
        <div class="expert-form">
            <table>
                <tr><th>Dog</th><th>Speed</th></tr>
                <tr><td>  Fast Eddie  </td><td>88</td></tr>
                <tr><td>"Lucky Joe"</td><td>79</td></tr>
                <tr><td>1. Thunder Bolt</td><td>85</td></tr>
                <tr><td>rocket dog</td><td>92</td></tr>
            </table>
        </div>
    </html>
    '''
    
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")
    
    # Check that names are properly cleaned and converted to uppercase
    expected_names = ['FAST EDDIE', 'LUCKY JOE', 'THUNDER BOLT', 'ROCKET DOG']
    
    for name in expected_names:
        assert name in extracted_data, f"Expected {name} to be in extracted data"
        assert 'speed_rating' in extracted_data[name]
        assert extracted_data[name]['expert_analysis'] is True


def test_empty_or_invalid_html(processor):
    """Test handling of empty or invalid HTML"""
    # Test empty HTML
    empty_html = '<html></html>'
    soup = BeautifulSoup(empty_html, 'html.parser')
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")
    assert extracted_data == {}
    
    # Test HTML with no relevant data
    irrelevant_html = '''
    <html>
        <body>
            <p>This is just regular text with no speed ratings.</p>
            <div>Some other content</div>
        </body>
    </html>
    '''
    soup = BeautifulSoup(irrelevant_html, 'html.parser')
    extracted_data = processor.extract_expert_form_data(soup, "http://example.com/race")
    assert extracted_data == {}

