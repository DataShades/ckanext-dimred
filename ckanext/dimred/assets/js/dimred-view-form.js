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
            this.featureColumnsSelect = $("#field-feature-columns");
            this.workloadPreflight = $("#dimred-workload-preflight");
            this.selectAllFeaturesButton = $("#dimred-select-all-features");
            this.clearFeaturesButton = $("#dimred-clear-features");
            this.resetButton = $("#dimred-reset-method-params");
            this.paramFields = this.container.find("[data-dimred-param]");
            this.defaults = this._parseDefaults(this.options.defaults || this.container.attr("data-module-defaults"));
            this.workloadProfiles = this._parseDefaults(this.workloadPreflight.attr("data-module-workload-profiles"));
            this.automaticFeaturesLabel = this.workloadPreflight.attr("data-module-automatic-features") || "automatic";

            if (!this.methodSelect.length || !this.paramsField.length) {
                return;
            }

            this.methodSelect.on("change", this._onMethodChange.bind(this));
            this.colorSelect.on("change", this._excludeColorFromFeatures.bind(this));
            this.featureColumnsSelect.on("change", this._updateWorkloadPreflight.bind(this));
            this.selectAllFeaturesButton.on("click", this._selectAllFeatures.bind(this));
            this.clearFeaturesButton.on("click", this._clearFeatures.bind(this));
            this.paramFields.on("change input", this._syncParams.bind(this));
            this.resetButton.on("click", this._resetCurrentMethod.bind(this));
            this.container.closest("form").on("submit", this._syncParams.bind(this));

            this._showMethodFields(this.methodSelect.val());
            this._updateWorkloadPreflight();
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
            this._updateWorkloadPreflight();
        },

        _resetCurrentMethod: function () {
            this._applyMethodDefaults(this.methodSelect.val());
            this._syncParams();
            this._updateWorkloadPreflight();
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
            if (!this.featureColumnsSelect.length) {
                return;
            }

            var colorBy = this.colorSelect.val();
            if (!colorBy) {
                return;
            }

            var selected = this.featureColumnsSelect.val() || [];
            this.featureColumnsSelect.val(selected.filter(function (column) {
                return column !== colorBy;
            })).trigger("change");
        },

        _selectAllFeatures: function () {
            if (!this.featureColumnsSelect.length) {
                return;
            }

            var colorBy = this.colorSelect.val();
            var values = this.featureColumnsSelect.find("option").map(function () {
                return $(this).val();
            }).get().filter(function (column) {
                return column !== colorBy;
            });
            this.featureColumnsSelect.val(values).trigger("change");
        },

        _clearFeatures: function () {
            if (this.featureColumnsSelect.length) {
                this.featureColumnsSelect.val([]).trigger("change");
            }
        },

        _updateWorkloadPreflight: function () {
            if (!this.workloadPreflight.length) {
                return;
            }

            var profile = this.workloadProfiles[this.methodSelect.val()];
            if (!profile) {
                this.workloadPreflight.prop("hidden", true);
                return;
            }

            var reference = profile.reference;
            var selectedFeatures = this.featureColumnsSelect.val() || [];
            var featureLabel = selectedFeatures.length ? selectedFeatures.length : this.automaticFeaturesLabel;

            this.workloadPreflight.prop("hidden", false);
            this.workloadPreflight.find("[data-dimred-workload-limit]").text(profile.max_rows);
            this.workloadPreflight.find("[data-dimred-workload-method]").text(profile.label);
            this.workloadPreflight.find("[data-dimred-workload-features]").text(featureLabel);
            this.workloadPreflight.find("[data-dimred-workload-reference-method]").text(profile.label);
            this.workloadPreflight.find("[data-dimred-workload-reference-rows]").text(reference.rows);
            this.workloadPreflight.find("[data-dimred-workload-reference-features]").text(reference.features);
            this.workloadPreflight.find("[data-dimred-workload-reference-time]").text(reference.wall_seconds.toFixed(2));
            this.workloadPreflight.find("[data-dimred-workload-reference-rss]").text(reference.peak_rss_mb);
            this.workloadPreflight.find("[data-dimred-workload-reference-params]").text(reference.params_text);
        },
    };
});
