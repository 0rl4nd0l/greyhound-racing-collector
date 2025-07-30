
# Greyhound Analysis Predictor - UI/UX Design Specification

This document outlines the UI/UX design for the Greyhound Analysis Predictor, a tool for visualizing and interacting with greyhound race data and predictions.

## 1. High-Level Design Philosophy

-   **Clarity and Information Density:** The UI should present complex information in a clear, digestible manner. Prioritize scannable layouts that allow users to quickly assess the state of upcoming races and model performance.
-   **Interactivity and Exploration:** Empower users to drill down into the data. All key data points should be interactive, leading to more detailed views or charts.
-   **Responsive and Modern:** The application will be fully responsive, providing a seamless experience across desktops, tablets, and mobile devices. A modern, clean aesthetic will be used throughout.
-   **Action-Oriented:** The design will guide users toward key actions, such as running predictions, viewing results, and managing the ML models.

## 2. Color Palette

-   **Primary:** `#2c5f2d` (Dark Green) - Used for headers, primary buttons, and key navigation elements.
-   **Secondary:** `#4caf50` (Green) - Used for secondary buttons, success states, and highlighting active elements.
-   **Accent:** `#ff9800` (Orange) - Used for calls to action, warnings, and to draw attention to important information.
-   **Neutral/Background:** `#f5f7fa` (Light Gray) - The primary background color for the application.
-   **Card/Surface:** `#ffffff` (White) - Background color for cards and modals.
-   **Text:** `#333333` (Dark Gray) - Primary text color.
-   **Text Light:** `#666666` (Gray) - For secondary text and labels.
-   **Borders:** `#e0e0e0` (Light Gray) - Used for table borders and card outlines.

## 3. Typography

-   **Primary Font:** 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
-   **Headings (h1, h2, h3):** Font weight 600
-   **Body Text:** Font weight 400
-   **Scale:**
    -   `h1`: 2.5rem
    -   `h2`: 2rem
    -   `h3`: 1.5rem
    -   `h4`: 1.2rem
    -   Body: 1rem (16px)
    -   Small: 0.875rem

## 4. Breakpoints for Responsive Design

-   **Mobile:** < 768px
-   **Tablet:** 768px - 1024px
-   **Desktop:** > 1024px

## 5. Interaction Patterns

-   **Loading States:** Use skeleton screens for initial page loads and spinners for in-component loading (e.g., refreshing a chart).
-   **Hover Effects:** Provide clear visual feedback on all interactive elements (buttons, links, cards) with subtle transitions (e.g., color changes, shadow elevation).
-   **Expand/Collapse:** Race cards and other detailed sections will be expandable to reveal more information without navigating to a new page.
-   **Bulk Actions:** The action bar will support bulk actions on selected race cards (e.g., "Run Predictions for Selected").
-   **Tooltips:** Provide additional context on hover for icons, chart elements, and truncated text.

## 6. Page Grid (Wireframe)

```
+----------------------------------------------------------------------------------+
| Header (Logo, App Title, Navigation Links, User/Status)                          |
+----------------------------------------------------------------------------------+
| Action Bar (Filters, Search, Bulk Actions: "Run All Predictions")                |
+----------------------------------------------------------------------------------+
| Main Content Area                                                                |
| +---------------------------------+ +------------------------------------------+ |
| | Race Cards Grid (Scrollable)    | | Sidebar/Status Drawer (Collapsible)      | |
| |                                 | |                                          | |
| | [Race Card 1 - Collapsed]       | | [ML Model Status]                        | |
| | [Race Card 2 - Collapsed]       | |                                          | |
| | [Race Card 3 - Expanded]        | | [System Health]                          | |
| |   - Runner Row 1                | |                                          | |
| |   - Runner Row 2                | | [Recent Activity]                        | |
| |   ...                           | |                                          | |
| +---------------------------------+ +------------------------------------------+ |
+----------------------------------------------------------------------------------+
| Footer (Copyright, Links)                                                        |
+----------------------------------------------------------------------------------+
```

## 7. Race Card Anatomy (Wireframe)

### Collapsed State

```
+----------------------------------------------------------------------------------+
| [Venue Name] - Race [Race Number] - [Distance]m - [Scheduled Time]               |
| [Status: Predicted/Unpredicted] [Expand Icon]                                    |
+----------------------------------------------------------------------------------+
```

### Expanded State

```
+----------------------------------------------------------------------------------+
| [Venue Name] - Race [Race Number] - [Distance]m - [Scheduled Time] [Collapse Icon] |
|----------------------------------------------------------------------------------|
| Runner | Confidence | Predicted Time | Win Prob | Place Prob | Odds (Live)        |
|--------|------------|----------------|----------|------------|--------------------|
| 1. Dog A | [|||||||---] 70% | 29.85s         | 35%      | 65%        | $3.50              |
| 2. Dog B | [|||||-----] 50% | 30.10s         | 20%      | 45%        | $5.00              |
| ...    | ...        | ...            | ...      | ...        | ...                |
|----------------------------------------------------------------------------------|
| [View Charts Button] [Run/Re-run Prediction Button]                              |
+----------------------------------------------------------------------------------+
```

## 8. Runner Row Layout

-   **Dog Name:** Includes box number.
-   **Confidence Bar:** A visual representation of the model's confidence in its prediction for this runner. The bar is color-coded based on confidence level (e.g., Green for high, Yellow for medium, Red for low).
-   **Predicted Time:** The model's predicted finish time.
-   **Win/Place Probabilities:** The probabilities assigned by the model.
-   **Live Odds:** Pulled from an integrated odds provider, updated in near real-time.

## 9. Chart Modal Layouts

-   **Trigger:** Clicking the "View Charts" button on an expanded race card.
-   **Layout:** A modal window with tabs for different charts.
-   **Charts:**
    -   **Win Probability Distribution:** Bar chart showing the win probability for each dog in the race.
    -   **Predicted Time Comparison:** Bar chart comparing the predicted finish times.
    -   **Historical Performance:** Line chart showing a dog's recent performance trends (e.g., finish position, speed over last 5 races).

## 10. CSS Framework

-   **Decision:** Vanilla CSS with SCSS variables.
-   **Rationale:**
    -   Given the specific and data-dense nature of the UI, a full framework like Bootstrap or Tailwind might add unnecessary overhead.
    -   Vanilla CSS with SCSS provides the flexibility to create a highly customized design while maintaining a structured and maintainable codebase.
    -   The existing `ml-dashboard.css` already utilizes CSS variables, which can be easily migrated to SCSS variables for more advanced features like mixins and functions.

