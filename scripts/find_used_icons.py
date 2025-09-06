#!/usr/bin/env python3

import glob
import os
import re


def find_fa_icons_in_file(file_path):
    """Extract FontAwesome icon classes from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return set()

    icons = set()

    # Pattern to find fa- classes (various formats)
    patterns = [
        r"fa-([a-z0-9-]+)",  # fa-icon-name
        r"fas\s+fa-([a-z0-9-]+)",  # fas fa-icon-name
        r"far\s+fa-([a-z0-9-]+)",  # far fa-icon-name
        r"fab\s+fa-([a-z0-9-]+)",  # fab fa-icon-name
        r"fal\s+fa-([a-z0-9-]+)",  # fal fa-icon-name
        r'"fa-([a-z0-9-]+)"',  # "fa-icon-name"
        r"'fa-([a-z0-9-]+)'",  # 'fa-icon-name'
        r'class="[^"]*fa-([a-z0-9-]+)[^"]*"',  # class="... fa-icon-name ..."
        r"class='[^']*fa-([a-z0-9-]+)[^']*'",  # class='... fa-icon-name ...'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            # Clean up the icon name
            icon_name = match.lower().strip()
            if (
                icon_name
                and not icon_name.startswith("fa-")
                and is_valid_icon(icon_name)
            ):
                icons.add(icon_name)

    return icons


def is_valid_icon(icon_name):
    """Check if an icon name is a valid FontAwesome icon."""
    # Filter out obviously invalid names
    if not icon_name or len(icon_name) < 2:
        return False

    # Filter out UUIDs and other non-FontAwesome patterns
    if len(icon_name) > 25:  # FA icons are typically short
        return False

    # Filter out patterns that look like UUIDs or IDs
    if re.match(r"^[a-f0-9]{8,}", icon_name):  # Looks like a hex ID
        return False

    if re.match(r"^[a-f0-9]{4}-[a-f0-9]{4}", icon_name):  # Looks like UUID pattern
        return False

    # Filter out size modifiers that aren't real icons
    size_modifiers = {"2x", "3x", "4x", "5x", "fw", "lg", "sm", "xs"}
    if icon_name in size_modifiers:
        return False

    # Filter out non-FA specific names
    invalid_patterns = [
        "champions-league",
        "europa-league",
        "competitions",
        "cup",
        "womens-euro-2025",
        "women-s-euro-2025-outright-9107609",
        "icon-name",
        "subset",
        "test",
    ]

    if icon_name in invalid_patterns:
        return False

    return True


def find_all_fa_icons():
    """Find all FontAwesome icons used in the project."""
    all_icons = set()

    # Search patterns for different file types
    search_patterns = [
        "static/**/*.js",
        "static/**/*.html",
        "templates/**/*.html",
        "**/*.html",
        "**/*.py",  # Some projects inline HTML in Python
    ]

    processed_files = []

    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            # Skip certain directories and files
            if any(
                skip in file_path
                for skip in ["node_modules", "venv", ".git", "__pycache__"]
            ):
                continue

            if file_path not in processed_files:
                processed_files.append(file_path)
                icons = find_fa_icons_in_file(file_path)
                if icons:
                    print(f"Found {len(icons)} icons in {file_path}: {sorted(icons)}")
                    all_icons.update(icons)

    return all_icons


def map_icon_to_import(icon_name):
    """Map icon name to FontAwesome import name."""
    # Convert kebab-case to camelCase with 'fa' prefix
    words = icon_name.split("-")
    camel_case = "fa" + "".join(word.capitalize() for word in words)
    return camel_case


def generate_import_statements(icons):
    """Generate import statements for the found icons."""
    if not icons:
        return ""

    # Group icons by likely package
    solid_icons = []
    regular_icons = []
    brand_icons = []

    # Common brand icons (this is a simplified list)
    brand_list = {
        "facebook",
        "twitter",
        "instagram",
        "linkedin",
        "github",
        "google",
        "apple",
        "microsoft",
        "amazon",
        "youtube",
        "whatsapp",
        "telegram",
    }

    # Common regular icons (this is a simplified list)
    regular_list = {
        "heart",
        "star",
        "bookmark",
        "calendar",
        "clock",
        "envelope",
        "file",
        "folder",
        "image",
        "user",
        "thumbs-up",
        "thumbs-down",
    }

    for icon in sorted(icons):
        import_name = map_icon_to_import(icon)

        if any(brand in icon for brand in brand_list):
            brand_icons.append(import_name)
        elif any(reg in icon for reg in regular_list):
            regular_icons.append(import_name)
        else:
            solid_icons.append(import_name)

    imports = []

    if solid_icons:
        imports.append(
            f"import {{ {', '.join(solid_icons)} }} from '@fortawesome/free-solid-svg-icons';"
        )

    if regular_icons:
        imports.append(
            f"import {{ {', '.join(regular_icons)} }} from '@fortawesome/free-regular-svg-icons';"
        )

    if brand_icons:
        imports.append(
            f"import {{ {', '.join(brand_icons)} }} from '@fortawesome/free-brands-svg-icons';"
        )

    return "\n".join(imports)


def main():
    print("🔍 Searching for FontAwesome icons in the project...")

    # Find all used icons
    icons = find_all_fa_icons()

    print(f"\n✅ Found {len(icons)} unique FontAwesome icons:")
    for icon in sorted(icons):
        print(f"  - fa-{icon}")

    if icons:
        print(f"\n📦 Import statements for fa-subset.js:")
        print(generate_import_statements(icons))

        print(f"\n🎯 Icon list for library.add():")
        import_names = [map_icon_to_import(icon) for icon in sorted(icons)]
        print(f"library.add({', '.join(import_names)});")
    else:
        print("\n⚠️ No FontAwesome icons found!")


if __name__ == "__main__":
    main()
