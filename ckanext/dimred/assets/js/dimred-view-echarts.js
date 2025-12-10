this.ckan.module("dimred-view-echarts", function ($) {
    "use strict";
    return {
        initialize: function () {
            var container = $("#dimred-js-render");
            var selectContainer = $("#dimred-color-select");

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
            var colorCandidatesRaw = prepareInfo.color_candidates;
            var colorCandidates = Array.isArray(colorCandidatesRaw) ? colorCandidatesRaw : [];
            var defaultColorBy = prepareInfo.color_by || "";
            var legacyColorValues = prepareInfo.color_values || [];

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
            var baseColor = palette[0];
            var missingColor = "#999999";

            var firstPoint = embedding[0] || [];
            var is3D = Array.isArray(firstPoint) && firstPoint.length >= 3;
            var dimNames = is3D ? ["x", "y", "z"] : ["x", "y"];
            var baseCoords = [];
            $.each(embedding, function (_, coords) {
                if (is3D) {
                    baseCoords.push([coords[0], coords[1], coords[2]]);
                } else {
                    baseCoords.push([coords[0], coords[1]]);
                }
            });

            var colorState = { name: null, kind: null };
            var candidateMap = {};
            $.each(colorCandidates, function (_, cand) {
                if (cand && cand.name) {
                    candidateMap[cand.name] = cand;
                }
            });

            var tooltipFormatter = function (params) {
                var coords = (params.data && params.data.__coords) || params.value || [];
                var lines = [];
                lines.push(dimNames[0] + ": " + coords[0]);
                lines.push(dimNames[1] + ": " + coords[1]);
                if (is3D && coords.length > 2) {
                    lines.push(dimNames[2] + ": " + coords[2]);
                }
                var colorVal = params.data ? params.data.__colorValue : null;
                if (colorState.name && colorVal !== undefined && colorVal !== null && colorVal !== "") {
                    lines.push(colorState.name + ": " + colorVal);
                }
                return lines.join("<br/>");
            };

            var baseSeries = {
                type: is3D ? "scatter3D" : "scatter",
                symbolSize: 6,
                data: $.map(baseCoords, function (coords) {
                    var c = (coords || []).slice(0);
                    return { value: c, __coords: c };
                }),
                encode: is3D ? { x: 0, y: 1, z: 2 } : { x: 0, y: 1 },
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
                    series: [baseSeries],
                    visualMap: [],
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
                    series: [baseSeries],
                    visualMap: [],
                };
            }

            var chart = echarts.init(container[0]);
            var baseOption = option;

            try {
                var cloneBaseOption = function () {
                    return $.extend(true, {}, baseOption);
                };

                chart.setOption(baseOption, true);

                var setColorState = function (name, kind) {
                    colorState.name = name || null;
                    colorState.kind = kind || null;
                };

                var applyUniformColor = function () {
                    var data = $.map(baseCoords, function (coords) {
                        var c = (coords || []).slice(0);
                        return { value: c, __coords: c, __colorValue: null, itemStyle: { color: baseColor } };
                    });
                    setColorState(null, null);
                    var newOption = cloneBaseOption();
                    newOption.visualMap = [];
                    newOption.series[0].data = data;
                    chart.setOption(newOption, true);
                };

                var applyCategorical = function (candidate) {
                    var values = Array.isArray(candidate.values) ? candidate.values : [];
                    var knownValues = Array.isArray(candidate.unique_values) ? candidate.unique_values : [];
                    var data = [];
                    var paletteIdx = 0;
                    var colorMap = {};

                    $.each(knownValues, function (_, v) {
                        if (v === null || v === undefined || v === "") {
                            return;
                        }
                        if (!colorMap[v]) {
                            colorMap[v] = palette[paletteIdx % palette.length];
                            paletteIdx += 1;
                        }
                    });

                    $.each(baseCoords, function (idx, coords) {
                        var label = idx < values.length ? values[idx] : null;
                        var itemColor = null;
                        if (label !== null && label !== undefined && label !== "") {
                            if (!colorMap[label]) {
                                colorMap[label] = palette[paletteIdx % palette.length];
                                paletteIdx += 1;
                            }
                            itemColor = colorMap[label];
                        }
                        var c = (coords || []).slice(0);
                        data.push({
                            value: c,
                            __coords: c,
                            __colorValue: label,
                            itemStyle: { color: itemColor || missingColor },
                        });
                    });

                    setColorState(candidate.name, "categorical");
                    var newOption = cloneBaseOption();
                    newOption.visualMap = [];
                    newOption.series[0].data = data;
                    chart.setOption(newOption, true);
                };

                var applyNumeric = function (candidate) {
                    var values = Array.isArray(candidate.values) ? candidate.values : [];
                    var numericValues = [];
                    $.each(values, function (_, val) {
                        if (typeof val === "number" && isFinite(val)) {
                            numericValues.push(val);
                        }
                    });

                    var minVal =
                        typeof candidate.min === "number" && isFinite(candidate.min)
                            ? candidate.min
                            : numericValues.length
                            ? Math.min.apply(null, numericValues)
                            : null;
                    var maxVal =
                        typeof candidate.max === "number" && isFinite(candidate.max)
                            ? candidate.max
                            : numericValues.length
                            ? Math.max.apply(null, numericValues)
                            : null;

                    if (minVal === null || maxVal === null) {
                        applyUniformColor();
                        return;
                    }

                    var data = [];
                    $.each(baseCoords, function (idx, coords) {
                        var rawVal = idx < values.length ? values[idx] : null;
                        var numericVal = typeof rawVal === "number" && isFinite(rawVal) ? rawVal : null;
                        var c = (coords || []).slice(0);
                        var valueWithColor = c.slice(0);
                        valueWithColor.push(numericVal);
                        data.push({
                            value: valueWithColor,
                            __coords: c,
                            __colorValue: numericVal,
                        });
                    });

                    setColorState(candidate.name, "numeric");
                    var newOption = cloneBaseOption();
                    newOption.visualMap = [
                        {
                            type: "continuous",
                            min: minVal,
                            max: maxVal,
                            dimension: is3D ? 3 : 2,
                            calculable: true,
                            inRange: {
                                color: ["#d2e9f7", "#0b62c3"],
                            },
                            seriesIndex: 0,
                        },
                    ];
                    newOption.series[0].data = data;
                    chart.setOption(newOption, true);
                };

                var applyColor = function (columnName) {
                    if (!columnName) {
                        applyUniformColor();
                        return;
                    }
                    var candidate = candidateMap[columnName];
                    if (!candidate) {
                        applyUniformColor();
                        return;
                    }
                    if (candidate.kind === "numeric") {
                        applyNumeric(candidate);
                    } else {
                        applyCategorical(candidate);
                    }
                };

                var buildSelector = function () {
                    if (!colorCandidates.length || !selectContainer || !selectContainer.length) {
                        return null;
                    }

                    var selectId = "dimred-color-select-input";
                    var label = $('<label class="dimred-color-select__label" for="' + selectId + '">Color by</label>');
                    var select = $('<select class="form-control dimred-color-select__control" id="' + selectId + '"></select>');
                    select.append('<option value="">None</option>');

                    $.each(colorCandidates, function (_, cand) {
                        if (!cand || !cand.name) {
                            return;
                        }
                        var optionEl = $("<option></option>").attr("value", cand.name).text(cand.name);
                        select.append(optionEl);
                    });

                    select.on("change", function () {
                        applyColor($(this).val());
                    });

                    selectContainer.empty().append(label).append(select);
                    return select;
                };

                var applyLegacyColor = function () {
                    var colorBy = prepareInfo.color_by || "";
                    var hasLabels = colorBy && legacyColorValues.length === baseCoords.length;
                    var data = [];
                    var colorMap = {};
                    var paletteIdx = 0;

                    $.each(baseCoords, function (idx, coords) {
                        var label = hasLabels ? legacyColorValues[idx] : null;
                        var color = baseColor;
                        if (label !== null && label !== undefined && label !== "") {
                            if (!colorMap[label]) {
                                colorMap[label] = palette[paletteIdx % palette.length];
                                paletteIdx += 1;
                            }
                            color = colorMap[label];
                        }
                        var c = (coords || []).slice(0);
                        data.push({
                            value: c,
                            __coords: c,
                            __colorValue: label,
                            itemStyle: { color: color },
                        });
                    });

                    setColorState(hasLabels ? colorBy : null, hasLabels ? "categorical" : null);
                    var newOption = cloneBaseOption();
                    newOption.visualMap = [];
                    newOption.series[0].data = data;
                    chart.setOption(newOption, true);
                };

                if (colorCandidates.length) {
                    var selector = buildSelector();
                    var initialValue = defaultColorBy && candidateMap[defaultColorBy] ? defaultColorBy : "";
                    applyColor(initialValue);
                    if (selector) {
                        selector.val(initialValue);
                    }
                } else {
                    applyLegacyColor();
                }

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
