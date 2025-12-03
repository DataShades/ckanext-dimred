ckan.module("dimred-view-form", function ($, _) {
    "use strict";
    return {
        options: {
            defaults: {},
        },

        initialize: function () {
            var attrDefaults = $("#field-method-params").attr("data-module-defaults");
            this.defaults = this._parseDefaults(this.options.defaults || attrDefaults);
            this.methodSelect = $("#field-method");
            this.paramsField = $("#field-method-params");

            if (!this.methodSelect.length || !this.paramsField.length) {
                return;
            }

            this.valueByMethod = {};
            this.currentMethod = this.methodSelect.val();
            this.currentDefault = this._stringifyDefault(this.currentMethod);
            this.valueByMethod[this.currentMethod] = this._normalizeJson(this.paramsField.val());

            this.methodSelect.on("change", this._onMethodChange.bind(this));
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

        _stringifyDefault: function (method) {
            var methodDefaults = this.defaults[method];
            return methodDefaults ? JSON.stringify(methodDefaults, null, 2) : "";
        },

        _normalizeJson: function (text) {
            var trimmed = (text || "").trim();
            if (!trimmed) {
                return "";
            }
            try {
                return JSON.stringify(JSON.parse(trimmed), null, 2);
            } catch (e) {
                return trimmed;
            }
        },

        _onMethodChange: function () {
            var newMethod = this.methodSelect.val();
            var currentValue = this._normalizeJson(this.paramsField.val());

            this.valueByMethod[this.currentMethod] = currentValue;

            var nextValue = this.valueByMethod[newMethod];
            if (!nextValue) {
                nextValue = this._stringifyDefault(newMethod);
            }

            if (nextValue) {
                this.paramsField.val(nextValue);
            }

            this.currentMethod = newMethod;
            this.currentDefault = this._stringifyDefault(newMethod);
        },
    };
});
