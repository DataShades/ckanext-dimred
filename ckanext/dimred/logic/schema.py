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
def dimred_start_preview_schema(
    not_empty: types.Validator,
    unicode_safe: types.Validator,
) -> types.Schema:
    """Validation schema for starting an asynchronous dimred preview."""
    return {
        "id": [not_empty, unicode_safe],
        "view_id": [not_empty, unicode_safe],
    }


@validator_args
def dimred_get_preview_status_schema(
    not_empty: types.Validator,
    unicode_safe: types.Validator,
) -> types.Schema:
    """Validation schema for retrieving an asynchronous preview status."""
    return {
        "id": [not_empty, unicode_safe],
        "view_id": [not_empty, unicode_safe],
        "job_id": [not_empty, unicode_safe],
    }


@validator_args
def dimred_get_dimred_color_values_schema(
    not_empty: types.Validator,
    unicode_safe: types.Validator,
) -> types.Schema:
    """Validation schema for loading one dimred color column."""
    return {
        "id": [not_empty, unicode_safe],
        "view_id": [not_empty, unicode_safe],
        "column": [not_empty, unicode_safe],
    }


@validator_args
def dimred_form_schema(  # noqa PLR0913
    ignore_empty: types.Validator,
    unicode_safe: types.Validator,
    dimred_allowed_method: types.Validator,
    dimred_feature_columns_list: types.Validator,
    dimred_display_fields_list: types.Validator,
    dimred_method_params_object: types.Validator,
    dimred_allowed_backend: types.Validator,
    dimred_n_components: types.Validator,
    ignore: types.Validator,
) -> types.Schema:
    """Validation schema for the dimred preview form."""
    return {
        "method": [ignore_empty, unicode_safe, dimred_allowed_method],
        "feature_columns": [ignore_empty, dimred_feature_columns_list],
        "color_by": [ignore_empty, unicode_safe],
        "display_fields": [ignore_empty, dimred_display_fields_list],
        "method_params": [ignore_empty, dimred_method_params_object],
        "render_backend": [ignore_empty, unicode_safe, dimred_allowed_backend],
        "n_components": [ignore_empty, dimred_n_components],
        "dimred_param_random_state": [ignore],
        "dimred_param_whiten": [ignore],
        "dimred_param_perplexity": [ignore],
        "dimred_param_n_neighbors": [ignore],
        "dimred_param_min_dist": [ignore],
    }


@validator_args
def dimred_export_embedding_schema(
    not_empty: types.Validator,
    unicode_safe: types.Validator,
) -> types.Schema:
    """Validation schema for exporting embedding."""
    return {
        "id": [not_empty, unicode_safe],
        "view_id": [not_empty, unicode_safe],
    }
