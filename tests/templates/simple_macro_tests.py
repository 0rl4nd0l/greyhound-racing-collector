#!/usr/bin/env python3
"""Simple template macro tests without pytest dependencies."""

import os
import sys

from jinja2 import Environment, FileSystemLoader

# Test configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")


def setup_jinja_env():
    """Create a Jinja2 environment for testing."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    return env


def get_macros_module():
    """Load the macros template."""
    env = setup_jinja_env()
    return env.get_template("components/macros.html").module


def test_alert_macro():
    """Test the alert macro."""
    macros = get_macros_module()

    # Test default values
    result = macros.alert("Test message")
    assert "alert alert-info" in result
    assert "Test message" in result
    assert "btn-close" in result
    assert "5000" in result  # default duration

    # Test with type
    result = macros.alert("Error message", "danger")
    assert "alert alert-danger" in result
    assert "Error message" in result

    # Test with custom duration
    result = macros.alert("Custom duration", "warning", 3000)
    assert "alert alert-warning" in result
    assert "3000" in result
    print("✓ Alert macro tests passed")


def test_spinner_macro():
    """Test the spinner macro."""
    macros = get_macros_module()

    result = macros.spinner()
    assert "spinner-border text-primary" in result
    assert "Loading..." in result
    assert "visually-hidden" in result
    print("✓ Spinner macro tests passed")


def test_badge_macro():
    """Test the badge macro."""
    macros = get_macros_module()

    # Test default values
    result = macros.badge("Test")
    assert "badge bg-primary" in result
    assert "Test" in result

    # Test with color
    result = macros.badge("Success", "success")
    assert "badge bg-success" in result
    assert "Success" in result
    print("✓ Badge macro tests passed")


def test_card_macro():
    """Test the card macro."""
    macros = get_macros_module()

    # Test basic card
    result = macros.card("Title", "Content")
    assert "card shadow-lg" in result
    assert "card-header bg-primary text-white" in result
    assert "Title" in result
    assert "Content" in result

    # Test card with icon
    result = macros.card("Title", "Content", icon="fas fa-test")
    assert "fas fa-test" in result

    # Test card without shadow
    result = macros.card("Title", "Content", shadow=False)
    assert "shadow-lg" not in result
    assert "card" in result
    print("✓ Card macro tests passed")


def test_modal_macro():
    """Test the modal macro."""
    macros = get_macros_module()

    # Test basic modal
    result = macros.modal("testModal", "Test Title", "Test Content")
    assert "modal fade" in result
    assert 'id="testModal"' in result
    assert "Test Title" in result
    assert "Test Content" in result

    # Test modal with size
    result = macros.modal("testModal", "Title", "Content", "xl")
    assert "modal-xl" in result
    print("✓ Modal macro tests passed")


def test_loading_spinner_macro():
    """Test the loading_spinner macro."""
    macros = get_macros_module()

    # Test default spinner
    result = macros.loading_spinner()
    assert "text-center py-4" in result
    assert "spinner-border text-primary" in result
    assert "Loading..." in result

    # Test custom text
    result = macros.loading_spinner("Please wait...")
    assert "Please wait..." in result
    print("✓ Loading spinner macro tests passed")


def test_pill_counter_macro():
    """Test the pill_counter macro."""
    macros = get_macros_module()

    # Test default counter
    result = macros.pill_counter(5)
    assert "badge bg-primary rounded-pill" in result
    assert "5" in result

    # Test with color
    result = macros.pill_counter(10, "success")
    assert "badge bg-success rounded-pill" in result
    assert "10" in result
    print("✓ Pill counter macro tests passed")


def test_progress_bar_macro():
    """Test the progress_bar macro."""
    macros = get_macros_module()

    # Test basic progress bar
    result = macros.progress_bar(75)
    assert "progress" in result
    assert "progress-bar bg-primary" in result
    assert "width: 75%" in result
    assert "75%" in result

    # Test animated progress bar
    result = macros.progress_bar(50, "warning", True)
    assert "progress-bar-striped progress-bar-animated" in result
    assert "bg-warning" in result
    assert "width: 50%" in result
    print("✓ Progress bar macro tests passed")


def test_macro_integration():
    """Test macro integration scenarios."""
    env = setup_jinja_env()

    # Test multiple macros together
    template_str = """
    {% import 'components/macros.html' as macros %}
    {{ macros.alert('Test alert', 'success') }}
    {{ macros.badge('Test badge', 'info') }}
    {{ macros.spinner() }}
    """
    template = env.from_string(template_str)
    result = template.render()

    assert "alert alert-success" in result
    assert "badge bg-info" in result
    assert "spinner-border" in result

    # Test macro with complex content using call blocks
    template_str = """
    {% import 'components/macros.html' as macros %}
    {% call macros.card('Complex Card', header_class='bg-success text-white', icon='fas fa-chart-bar') %}
        <div class="row">
            <div class="col-md-6">Column 1</div>
            <div class="col-md-6">Column 2</div>
        </div>
    {% endcall %}
    """
    template = env.from_string(template_str)
    result = template.render()

    assert "Complex Card" in result
    assert "bg-success text-white" in result
    assert "fas fa-chart-bar" in result
    assert "Column 1" in result
    assert "Column 2" in result
    print("✓ Macro integration tests passed")


def run_all_tests():
    """Run all macro tests."""
    print("Running Jinja macro tests...")
    print("-" * 40)

    try:
        test_alert_macro()
        test_spinner_macro()
        test_badge_macro()
        test_card_macro()
        test_modal_macro()
        test_loading_spinner_macro()
        test_pill_counter_macro()
        test_progress_bar_macro()
        test_macro_integration()

        print("-" * 40)
        print("✓ All macro tests passed successfully!")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
