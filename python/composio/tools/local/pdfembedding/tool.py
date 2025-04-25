"""
PDF embedding tool for composio
"""

import typing as t
from composio.tools.base.local import LocalAction, LocalTool

from .actions import PDFEmbedding



class PDFEmbedding(LocalTool, autoload=True):
    """PDF embedding tool for local usage"""

    logo="https://github.com/imkrish7/composio/blob/e4110eb9c7db049fc27a399dc6db85d821306f09/python/docs/imgs/logos/pdfembedder.png"

    @classmethod
    def actions(cls)-> t.List[t.Type[LocalAction]]:
        """Return the list of actions"""
        return [PDFEmbedding]

