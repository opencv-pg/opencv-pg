import logging
from pathlib import Path

from opencv_pg import env
from jinja2.exceptions import TemplateNotFound

log = logging.getLogger(__name__)

ROOT = Path(__file__)
RENDERED_DIR = ROOT.parent.joinpath("rendered_docs/")


def _create_rendered_docs():
    """Create rendered_docs folder if it doesn't exist"""
    if not RENDERED_DIR.exists():
        RENDERED_DIR.mkdir()
        log.info("Created %s", RENDERED_DIR)


def render_local_doc(folder, doc_fname):
    """Renders template into a local file

    Normally we would render content in the webview with `setHtml`, but there
    seems to be a bug which doesn't load local resources when that method is
    used. It works properly when loaded from a local file.
    """
    try:
        template = env.get_template(doc_fname)
        html = template.render()
    except TemplateNotFound:
        template = env.get_template("error.html")
        html = template.render(error=f"Template: {doc_fname} not found :(")

    path = folder.joinpath(doc_fname)
    with open(path, "w") as fout:
        fout.write(html)
        log.debug("Wrote Doc: %s", path)


# Create rendered docs folder if it doesn't exist
_create_rendered_docs()
