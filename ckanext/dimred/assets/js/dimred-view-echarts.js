this.ckan.module("dimred-view-echarts", function ($) {
    "use strict";
    return {
        initialize: function () {
            var container = $("#dimred-js-render");
            if (!container.length) {
                console.error("dimred-view-echarts: container not found");
                return;
            }
            if (!window.echarts) {
                console.error("dimred-view-echarts: echarts not available");
                return;
            }

            var rawEmbedding = container.attr("data-embedding");
            var rawMeta = container.attr("data-meta");
            var embedding, meta;

            try {
                embedding = typeof rawEmbedding === "string" ? JSON.parse(rawEmbedding) : rawEmbedding;
                meta = rawMeta ? JSON.parse(rawMeta) : {};
            } catch (e) {
                console.error("dimred-view-echarts: failed to parse embedding/meta", e);
                container.text("Failed to render embedding (parse error).");
                return;
            }

            if (!embedding || !embedding.length) {
                container.text("No embedding data available.");
                return;
            }

            var prepareInfo = meta.prepare_info || {};
            var colorBy = prepareInfo.color_by;
            var colorValues = prepareInfo.color_values || [];

            var palette = [
                "#5470c6",
                "#91cc75",
                "#fac858",
                "#ee6666",
                "#73c0de",
                "#3ba272",
                "#fc8452",
                "#9a60b4",
                "#ea7ccc",
            ];
            var colorMap = {};
            var paletteIdx = 0;

            var firstPoint = embedding[0] || [];
            var is3D = Array.isArray(firstPoint) && firstPoint.length >= 3;
            var dimNames = is3D ? ["x", "y", "z"] : ["x", "y"];
            var points = [];
            $.each(embedding, function (idx, coords) {
                var value = is3D ? [coords[0], coords[1], coords[2]] : [coords[0], coords[1]];
                var label = colorValues.length === embedding.length ? colorValues[idx] : null;
                var color = null;
                if (label !== null && label !== undefined && label !== "") {
                    if (!colorMap[label]) {
                        colorMap[label] = palette[paletteIdx % palette.length];
                        paletteIdx += 1;
                    }
                    color = colorMap[label];
                }
                points.push({ value: value, label: label, itemStyle: color ? { color: color } : undefined });
            });

            var chart = echarts.init(container[0]);
            try {
                var tooltipFormatter = function (params) {
                    var v = params.value;
                    var lbl = params.data.label;
                    var text = dimNames[0] + ": " + v[0] + "<br/>" + dimNames[1] + ": " + v[1];
                    if (is3D) {
                        text += "<br/>" + dimNames[2] + ": " + v[2];
                    }
                    if (colorBy && lbl !== null && lbl !== undefined) {
                        text += "<br/>" + colorBy + ": " + lbl;
                    }
                    return text;
                };

                var option;
                if (is3D) {
                    option = {
                        tooltip: {
                            trigger: "item",
                            formatter: tooltipFormatter,
                        },
                        xAxis3D: { type: "value", name: dimNames[0] },
                        yAxis3D: { type: "value", name: dimNames[1] },
                        zAxis3D: { type: "value", name: dimNames[2] },
                        grid3D: {
                            viewControl: {
                                projection: "perspective",
                                rotateSensitivity: 1,
                                zoomSensitivity: 1,
                                panSensitivity: 1,
                            },
                        },
                        series: [
                            {
                                type: "scatter3D",
                                symbolSize: 6,
                                data: points,
                            },
                        ],
                    };
                } else {
                    option = {
                        tooltip: {
                            trigger: "item",
                            formatter: tooltipFormatter,
                        },
                        xAxis: { type: "value", name: dimNames[0] },
                        yAxis: { type: "value", name: dimNames[1] },
                        dataZoom: [
                            { type: "inside", filterMode: "none" },
                            { type: "slider", filterMode: "none" },
                        ],
                        series: [
                            {
                                type: "scatter",
                                symbolSize: 6,
                                data: points,
                            },
                        ],
                    };
                }

                chart.setOption(option);
                $(window).on("resize", function () {
                    chart.resize();
                });
            } catch (err) {
                console.error("dimred-view-echarts: failed to render chart", err);
                container.text("Failed to render embedding (chart error).");
            }
        },
    };
});
