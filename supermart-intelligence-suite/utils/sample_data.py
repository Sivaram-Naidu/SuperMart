"""Static sample data for the SuperMart SIS dashboard.

All data here is placeholder data. Replace the functions below with
calls to real data sources / ML model endpoints in the future; the UI
will not need to change.
"""

# ---------- KPI cards ----------

KPI_CARDS = [
    {
        "title": "Total Customers",
        "value": "48,920",
        "trend": "+12.4%",
        "trend_up": True,
        "icon": "users",
        "color": "#1976D2",
    },
    {
        "title": "Monthly Revenue",
        "value": "$2.84M",
        "trend": "+8.1%",
        "trend_up": True,
        "icon": "revenue",
        "color": "#22C55E",
    },
    {
        "title": "Daily Transactions",
        "value": "12,340",
        "trend": "-2.3%",
        "trend_up": False,
        "icon": "transactions",
        "color": "#F59E0B",
    },
    {
        "title": "Prediction Accuracy",
        "value": "94.7%",
        "trend": "+1.2%",
        "trend_up": True,
        "icon": "accuracy",
        "color": "#8B5CF6",
    },
]

# ---------- Revenue performance (weekly bar chart) ----------

REVENUE_WEEKS = [
    "Week 1",
    "Week 2",
    "Week 3",
    "Week 4",
    "Week 5",
    "Week 6",
    "Week 7",
    "Week 8",
]

REVENUE_ONLINE = [320, 410, 380, 460, 490, 520, 540, 610]
REVENUE_STORE = [280, 330, 360, 390, 420, 450, 470, 530]

# ---------- Customer retention heatmap ----------

RETENTION_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RETENTION_HOURS = [
    "6am",
    "9am",
    "12pm",
    "3pm",
    "6pm",
    "9pm",
    "12am",
]
# rows = weekdays, cols = hours (values 0-100 retention score)
RETENTION_MATRIX = [
    [12, 34, 55, 40, 28, 18, 8],
    [15, 42, 68, 52, 35, 22, 10],
    [18, 48, 75, 60, 42, 28, 14],
    [22, 52, 80, 66, 48, 32, 16],
    [30, 60, 88, 78, 62, 45, 24],
    [40, 72, 95, 90, 80, 60, 35],
    [28, 55, 82, 70, 55, 38, 20],
]

# ---------- Recent intelligence logs ----------

INTELLIGENCE_LOGS = [
    {
        "event": "Churn model retrained successfully",
        "module": "Customer Churn",
        "status": "Success",
        "timestamp": "2023-11-14 09:42",
    },
    {
        "event": "New market basket rules generated",
        "module": "Market Basket",
        "status": "System",
        "timestamp": "2023-11-14 09:15",
    },
    {
        "event": "Ad campaign budget threshold exceeded",
        "module": "Ad Optimization",
        "status": "Warning",
        "timestamp": "2023-11-14 08:50",
    },
    {
        "event": "Sentiment batch analysis completed",
        "module": "Sentiment Analysis",
        "status": "Success",
        "timestamp": "2023-11-14 08:22",
    },
    {
        "event": "Deep learning training job queued",
        "module": "Deep Learning",
        "status": "System",
        "timestamp": "2023-11-14 07:58",
    },
    {
        "event": "Segmentation clusters refreshed",
        "module": "Customer Segmentation",
        "status": "Success",
        "timestamp": "2023-11-14 07:30",
    },
]

# ---------- Navigation ----------

NAV_ITEMS = [
    ("Home", "home"),
    ("Customer Churn", "churn"),
    ("Customer Segmentation", "segmentation"),
    ("Market Basket", "basket"),
    ("Ad Optimization", "ads"),
    ("Sentiment Analysis", "sentiment"),
    ("Deep Learning", "deep_learning"),
]

SUPPORT_ITEMS = [
    ("About", "about"),
    ("Help", "help"),
]
