import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Dashboard configuration
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "30"))  # seconds
DEFAULT_BUILD_CYCLE = os.getenv("DEFAULT_BUILD_CYCLE", "latest")
ENABLE_AUTO_REFRESH = os.getenv("ENABLE_AUTO_REFRESH", "true").lower() == "true"

# Theme configuration
THEME_COLORS = {
    'primary': '#3B82F6',
    'secondary': '#10B981',
    'accent': '#8B5CF6',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'dark': '#1F2937',
    'light': '#F9FAFB'
}

# Data limits
MAX_BUILDS_DISPLAY = 10
MAX_TESTS_DISPLAY = 1000
HEATMAP_SIZE = (10, 15)  # rows, columns

# Feature flags
ENABLE_AI_RECOMMENDATIONS = True
ENABLE_EXPORT = True
ENABLE_ALERTS = True