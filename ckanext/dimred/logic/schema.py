from ckan import types
from ckan.logic.schema import validator_args


@validator_args
def dimred_get_dimred_preview_schema(
    not_empty: types.Validator,
    unicode_safe: types.Validator,
) -> types.Schema:
    """Validation schema for the dimred_get_dimred_preview action."""
    return {
        "id": [not_empty, unicode_safe],
        "view_id": [not_empty, unicode_safe],
    }


@validator_args
def dimred_form_schema(
    ignore_empty: types.Validator,
    unicode_safe: types.Validator,
    dimred_allowed_method: types.Validator,
    dimred_feature_columns_list: types.Validator,
    dimred_method_params_object: types.Validator,
) -> types.Schema:
    """Validation schema for the dimred preview form."""
    return {
        "method": [ignore_empty, unicode_safe, dimred_allowed_method],
        "feature_columns": [ignore_empty, dimred_feature_columns_list],
        "color_by": [ignore_empty, unicode_safe],
        "method_params": [ignore_empty, dimred_method_params_object],
    }
