"""Dream worker — scheduled maintenance jobs (decay, duplicate detection, promotion).

Phase 0: an idle loop with structured logging so the container can run.
Phase 2 will register APScheduler jobs for decay/duplicates/promotion.
"""

__version__ = "0.0.1"
