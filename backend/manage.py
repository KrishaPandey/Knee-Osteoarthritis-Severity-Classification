#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    # Ensure the backend folder is in the system path if not automatically included
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

    # Set the default settings module for the 'django' program.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

    try:
        # Import Django's management utility and run commands
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Run the command line interface for Django admin tasks
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()