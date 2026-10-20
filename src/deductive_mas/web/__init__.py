"""The tutor in a browser: `dmas serve`.

service.py runs the terminal session's rules for many learners at once,
views.py decides what of a result the browser may see, server.py speaks
HTTP, static/ is the front end. Nothing here diagnoses anything.
"""

from .server import make_server, serve
from .service import TutorService, WebError

__all__ = ["TutorService", "WebError", "make_server", "serve"]
