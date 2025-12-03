from __future__ import annotations

from ckan import types
from ckan.plugins import toolkit as tk


def dimred_get_dimred_preview(context: types.Context, data_dict: types.dict[str, str]) -> dict[str, bool]:
    """Authorize access to the dimred preview action."""
    tk.check_access("resource_show", context, {"id": data_dict["id"]})
    tk.check_access("resource_view_show", context, {"id": data_dict["view_id"]})
    return {"success": True}
