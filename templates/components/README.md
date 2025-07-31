# Template Components

This directory contains reusable Jinja2 template components for the Greyhound Racing Collector application.

## Structure

```
templates/components/
├── README.md           # This documentation
├── macros.html         # Reusable Jinja2 macros
├── navbar.html         # Navigation bar component
└── sidebar.html        # System status sidebar component
```

## Components

### Navbar (`navbar.html`)
Complete navigation bar with dropdowns for different sections of the application.

**Usage:**
```jinja2
{% include 'components/navbar.html' %}
```

### Sidebar (`sidebar.html`)
System status sidebar with logs, model metrics, and system health information.

**Usage:**
```jinja2
{% include 'components/sidebar.html' %}
```

### Macros (`macros.html`)
Collection of reusable UI components as Jinja2 macros.

**Usage:**
```jinja2
{% import 'components/macros.html' as macros %}
```

## Available Macros

### Alert
Creates Bootstrap alert components with auto-dismiss functionality.

```jinja2
{{ macros.alert('Success message', 'success', 3000) }}
```

**Parameters:**
- `message` (string): Alert message text
- `type` (string, default='info'): Bootstrap alert type (success, danger, warning, info)
- `duration` (int, default=5000): Auto-dismiss duration in milliseconds

### Badge
Creates Bootstrap badge components.

```jinja2
{{ macros.badge('New', 'primary') }}
```

**Parameters:**
- `text` (string): Badge text
- `color` (string, default='primary'): Bootstrap color class

### Pill Counter
Creates rounded pill-style badges for counters.

```jinja2
{{ macros.pill_counter(5, 'success') }}
```

**Parameters:**
- `count` (int): Counter value
- `color` (string, default='primary'): Bootstrap color class

### Spinner
Creates a basic loading spinner.

```jinja2
{{ macros.spinner() }}
```

### Loading Spinner
Creates a centered loading spinner with text.

```jinja2
{{ macros.loading_spinner('Please wait...') }}
```

**Parameters:**
- `text` (string, default='Loading...'): Loading message

### Progress Bar
Creates Bootstrap progress bars with optional animation.

```jinja2
{{ macros.progress_bar(75, 'success', true) }}
```

**Parameters:**
- `percentage` (int): Progress percentage (0-100)
- `color` (string, default='primary'): Bootstrap color class
- `animated` (boolean, default=false): Enable striped animation

### Card
Creates Bootstrap card components with header and body.

```jinja2
<!-- Simple card -->
{{ macros.card('Title', 'Content', 'bg-primary text-white', 'fas fa-icon') }}

<!-- Card with complex content using call blocks -->
{% call macros.card('Title', header_class='bg-success text-white', icon='fas fa-chart') %}
    <div class="row">
        <div class="col-md-6">Column 1</div>
        <div class="col-md-6">Column 2</div>
    </div>
{% endcall %}
```

**Parameters:**
- `title` (string): Card title
- `content` (string, default=''): Card content (ignored if using call blocks)
- `header_class` (string, default='bg-primary text-white'): Header CSS classes
- `icon` (string, default=''): Font Awesome icon class for title
- `shadow` (boolean, default=true): Enable card shadow

### Modal
Creates Bootstrap modal components.

```jinja2
{{ macros.modal('myModal', 'Modal Title', 'Modal content', 'xl') }}
```

**Parameters:**
- `id` (string): Modal ID
- `title` (string): Modal title
- `content` (string): Modal body content
- `size` (string, default=''): Modal size (sm, lg, xl)

## Testing

Unit tests for all macros are available in `tests/templates/simple_macro_tests.py`.

Run tests with:
```bash
source venv/bin/activate
python tests/templates/simple_macro_tests.py
```

## Refactored Templates

The following templates have been refactored to use these components:

- `ml_dashboard.html` - Uses macros for alerts, badges, and improved structure
- `interactive_races.html` - Uses card macro for cleaner layout
- `base.html` - Uses navbar and sidebar components

## Benefits

1. **Reusability**: Components can be shared across multiple templates
2. **Consistency**: Standardized UI components ensure consistent design
3. **Maintainability**: Changes to components automatically apply everywhere
4. **Testability**: Individual components can be unit tested
5. **Readability**: Templates are cleaner and more focused on content structure
