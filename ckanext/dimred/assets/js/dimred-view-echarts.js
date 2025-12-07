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

            var points = [];
            $.each(embedding, function (idx, coords) {
                var x = coords[0];
                var y = coords[1];
                var label = colorValues.length === embedding.length ? colorValues[idx] : null;
                var color = null;
                if (label !== null && label !== undefined && label !== "") {
                    if (!colorMap[label]) {
                        colorMap[label] = palette[paletteIdx % palette.length];
                        paletteIdx += 1;
                    }
                    color = colorMap[label];
                }
                points.push({ value: [x, y], label: label, itemStyle: color ? { color: color } : undefined });
            });

            var chart = echarts.init(container[0]);
            try {
                var option = {
                    tooltip: {
                        trigger: "item",
                        formatter: function (params) {
                            var v = params.value;
                            var lbl = params.data.label;
                            var text = "x: " + v[0] + "<br/>y: " + v[1];
                            if (colorBy && lbl !== null && lbl !== undefined) {
                                text += "<br/>" + colorBy + ": " + lbl;
                            }
                            return text;
                        },
                    },
                    xAxis: { type: "value" },
                    yAxis: { type: "value" },
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
