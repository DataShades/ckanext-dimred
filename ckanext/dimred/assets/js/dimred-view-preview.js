this.ckan.module("dimred-view-preview", function ($) {
    "use strict";
    var pollInterval = 1000;
    var maxAttempts = 60;

    return {
        initialize: function () {
            var container = $("#dimred-preview-status");
            var statusUrl = container.attr("data-status-url");
            var resourceId = container.attr("data-resource-id");
            var viewId = container.attr("data-view-id");
            var jobId = container.attr("data-job-id");
            var attempts = 0;

            if (!container.length || !statusUrl || !resourceId || !viewId || !jobId) {
                return;
            }

            var showError = function (message) {
                container
                    .removeClass("alert-info")
                    .addClass("alert-danger")
                    .text(message || "Dimred preview failed.");
            };

            var poll = function () {
                attempts += 1;
                if (attempts > maxAttempts) {
                    showError("Dimred preview is taking longer than expected. Reload the page to check its status.");
                    return;
                }

                $.ajax({
                    url: statusUrl,
                    dataType: "json",
                    data: {
                        id: resourceId,
                        view_id: viewId,
                        job_id: jobId,
                    },
                })
                    .done(function (response) {
                        var result = response && response.success ? response.result : null;
                        if (!result) {
                            showError("Unable to check dimred preview status.");
                            return;
                        }
                        if (result.status === "ready") {
                            window.location.reload();
                            return;
                        }
                        if (result.status === "failed") {
                            showError(result.error || "Dimred preview failed.");
                            return;
                        }
                        if (result.status === "running") {
                            container.text("Generating dimensionality reduction preview…");
                        }
                        window.setTimeout(poll, pollInterval);
                    })
                    .fail(function () {
                        showError("Unable to check dimred preview status.");
                    });
            };

            window.setTimeout(poll, pollInterval);
        },
    };
});
