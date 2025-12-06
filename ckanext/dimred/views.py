from __future__ import annotations

from flask import Blueprint, Response

import ckan.plugins.toolkit as tk

from ckanext.dimred.exception import DimredError

dimred = Blueprint("dimred", __name__)


@dimred.route("/dimred/export/<resource_id>/<view_id>")
def export_embedding(resource_id: str, view_id: str):
    context = {"user": tk.g.user or tk.g.author, "auth_user_obj": tk.g.userobj}
    try:
        result = tk.get_action("dimred_export_embedding")(context, {"id": resource_id, "view_id": view_id})
    except tk.NotAuthorized:
        return tk.abort(403, tk._("Not authorized"))
    except tk.ValidationError as err:
        return tk.abort(400, str(err))
    except DimredError as err:
        return tk.abort(400, str(err))

    headers = {
        "Content-Type": result.get("content_type", "text/csv; charset=utf-8"),
        "Content-Disposition": f'attachment; filename="{result["filename"]}"',
    }
    return Response(result["content"], headers=headers)
