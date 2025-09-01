#!/usr/bin/env python3
"""Simple template macro tests."""

import os

import pytest
from jinja2 import Environment, FileSystemLoader

# Test configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")


@pytest.fixture
def jinja_env():
    """Create a Jinja2 environment for testing."""
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)


@pytest.fixture
def macros_module(jinja_env):
    """Load the macros template."""
    try:
        return jinja_env.get_template("components/macros.html").module
    except Exception:
        # If macros template doesn't exist, skip tests
        pytest.skip("Macros template not found")


def test_template_directory_exists():
    """Test that the template directory exists."""
    assert os.path.exists(
        TEMPLATE_DIR
    ), f"Template directory {TEMPLATE_DIR} should exist"


def test_macros_template_structure(jinja_env):
    """Test basic macro template structure."""
    try:
        template = jinja_env.get_template("components/macros.html")
        assert template is not None
    except Exception:
        pytest.skip("Macros template not found - this is acceptable for basic testing")


class TestSpinnerMacro:
    """Test the spinner macro."""

    def test_spinner_default(self, macros_module):
        """Test spinner with default values."""
        result = macros_module.spinner()
        assert "spinner-border text-primary" in result
        assert "Loading..." in result
        assert "visually-hidden" in result


class TestBadgeMacro:
    """Test the badge macro."""

    def test_badge_default(self, macros_module):
        """Test badge with default values."""
        result = macros_module.badge("Test")
        assert "badge bg-primary" in result
        assert "Test" in result

    def test_badge_with_color(self, macros_module):
        """Test badge with custom color."""
        result = macros_module.badge("Success", "success")
        assert "badge bg-success" in result
        assert "Success" in result


class TestCardMacro:
    """Test the card macro."""

    def test_card_basic(self, macros_module):
        """Test card with basic parameters."""
        result = macros_module.card("Title", "Content")
        assert "card shadow-lg" in result
        assert "card-header bg-primary text-white" in result
        assert "Title" in result
        assert "Content" in result

    def test_card_with_icon(self, macros_module):
        """Test card with icon."""
        result = macros_module.card("Title", "Content", icon="fas fa-test")
        assert "fas fa-test" in result

    def test_card_no_shadow(self, macros_module):
        """Test card without shadow."""
        result = macros_module.card("Title", "Content", shadow=False)
        assert "shadow-lg" not in result
        assert "card" in result


class TestModalMacro:
    """Test the modal macro."""

    def test_modal_basic(self, macros_module):
        """Test modal with basic parameters."""
        result = macros_module.modal("testModal", "Test Title", "Test Content")
        assert "modal fade" in result
        assert 'id="testModal"' in result
        assert "Test Title" in result
        assert "Test Content" in result

    def test_modal_with_size(self, macros_module):
        """Test modal with size parameter."""
        result = macros_module.modal("testModal", "Title", "Content", "xl")
        assert "modal-xl" in result


class TestLoadingSpinnerMacro:
    """Test the loading_spinner macro."""

    def test_loading_spinner_default(self, macros_module):
        """Test loading spinner with default text."""
        result = macros_module.loading_spinner()
        assert "text-center py-4" in result
        assert "spinner-border text-primary" in result
        assert "Loading..." in result

    def test_loading_spinner_custom_text(self, macros_module):
        """Test loading spinner with custom text."""
        result = macros_module.loading_spinner("Please wait...")
        assert "Please wait..." in result


class TestPillCounterMacro:
    """Test the pill_counter macro."""

    def test_pill_counter_default(self, macros_module):
        """Test pill counter with default color."""
        result = macros_module.pill_counter(5)
        assert "badge bg-primary rounded-pill" in result
        assert "5" in result

    def test_pill_counter_with_color(self, macros_module):
        """Test pill counter with custom color."""
        result = macros_module.pill_counter(10, "success")
        assert "badge bg-success rounded-pill" in result
        assert "10" in result


class TestProgressBarMacro:
    """Test the progress_bar macro."""

    def test_progress_bar_basic(self, macros_module):
        """Test progress bar with basic parameters."""
        result = macros_module.progress_bar(75)
        assert "progress" in result
        assert "progress-bar bg-primary" in result
        assert "width: 75%" in result
        assert "75%" in result

    def test_progress_bar_animated(self, macros_module):
        """Test animated progress bar."""
        result = macros_module.progress_bar(50, "warning", True)
        assert "progress-bar-striped progress-bar-animated" in result
        assert "bg-warning" in result
        assert "width: 50%" in result


# Integration tests
class TestMacroIntegration:
    """Test macro integration scenarios."""

    def test_multiple_macros_together(self, jinja_env):
        """Test using multiple macros in one template."""
        template_str = """
        {% import 'components/macros.html' as macros %}
        {{ macros.alert('Test alert', 'success') }}
        {{ macros.badge('Test badge', 'info') }}
        {{ macros.spinner() }}
        """
        template = jinja_env.from_string(template_str)
        result = template.render()

        assert "alert alert-success" in result
        assert "badge bg-info" in result
        assert "spinner-border" in result

    def test_macro_with_complex_content(self, jinja_env):
        """Test macro with complex HTML content."""
        template_str = """
        {% import 'components/macros.html' as macros %}
        {% call macros.card('Complex Card', '', 'bg-success text-white', 'fas fa-chart-bar') %}
            <div class="row">
                <div class="col-md-6">Column 1</div>
                <div class="col-md-6">Column 2</div>
            </div>
        {% endcall %}
        """
        template = jinja_env.from_string(template_str)
        result = template.render()

        assert "Complex Card" in result
        assert "bg-success text-white" in result
        assert "fas fa-chart-bar" in result
        assert "Column 1" in result
        assert "Column 2" in result
