# Entry point — delegates to app_enhanced.py
# docker-compose references app.py; this ensures both work.
from app_enhanced import main
main()
