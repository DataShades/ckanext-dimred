ckan.module("dimred-view-form", function ($) {
    "use strict";
    return {
        options: {
            defaults: {},
        },

        initialize: function () {
            this.container = $(this.el);
            this.methodSelect = $("#field-method");
            this.componentsField = $("#field-n-components");
            this.paramsField = $("#field-method-params");
            this.colorSelect = $("#field-color_by");
            this.resetButton = $("#dimred-reset-method-params");
            this.paramFields = this.container.find("[data-dimred-param]");
            this.defaults = this._parseDefaults(this.options.defaults || this.container.attr("data-module-defaults"));

            if (!this.methodSelect.length || !this.paramsField.length) {
                return;
            }

            this.methodSelect.on("change", this._onMethodChange.bind(this));
            this.colorSelect.on("change", this._excludeColorFromFeatures.bind(this));
            this.paramFields.on("change input", this._syncParams.bind(this));
            this.resetButton.on("click", this._resetCurrentMethod.bind(this));
            this.container.closest("form").on("submit", this._syncParams.bind(this));

            this._showMethodFields(this.methodSelect.val());
        },

        _parseDefaults: function (defaults) {
            if (typeof defaults === "string") {
                try {
                    return JSON.parse(defaults);
                } catch (e) {
                    return {};
                }
            }
            return defaults || {};
        },

        _onMethodChange: function () {
            var method = this.methodSelect.val();
            this._applyMethodDefaults(method);
            this._showMethodFields(method);
            this._syncParams();
        },

        _resetCurrentMethod: function () {
            this._applyMethodDefaults(this.methodSelect.val());
            this._syncParams();
        },

        _applyMethodDefaults: function (method) {
            var defaults = this.defaults[method] || {};
            this._applyParams(defaults);
            this.componentsField.val(defaults.n_components === undefined ? "" : defaults.n_components);
        },

        _applyParams: function (params) {
            this.paramFields.each(function () {
                var field = $(this);
                var value = params[field.data("dimred-param")];
                if (field.attr("type") === "checkbox") {
                    field.prop("checked", Boolean(value));
                } else {
                    field.val(value === undefined || value === null ? "" : value);
                }
            });
        },

        _showMethodFields: function (method) {
            this.container.find("[data-dimred-method]").each(function () {
                $(this).prop("hidden", $(this).data("dimred-method") !== method);
            });
        },

        _syncParams: function () {
            var method = this.methodSelect.val();
            var allowed = this.defaults[method] || {};
            var params = {};

            this.paramFields.each(function () {
                var field = $(this);
                var name = field.data("dimred-param");
                if (!(name in allowed)) {
                    return;
                }

                if (field.attr("type") === "checkbox") {
                    params[name] = field.is(":checked");
                    return;
                }

                var value = field.val();
                if (value !== "") {
                    params[name] = Number(value);
                }
            });

            this.paramsField.val(JSON.stringify(params));
        },

        _excludeColorFromFeatures: function () {
            var colorBy = this.colorSelect.val();
            if (!colorBy) {
                return;
            }

            $("#feature-columns-list input").filter(function () {
                return $(this).val() === colorBy;
            }).prop("checked", false);
        },
    };
});
