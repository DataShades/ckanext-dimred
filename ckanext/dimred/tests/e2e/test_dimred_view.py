from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import pytest
from playwright.sync_api import expect
from werkzeug.datastructures import FileStorage

from ckan.plugins import toolkit as tk
from ckan.tests import factories
from ckan.tests.helpers import call_action

COLOR_ACTION = "dimred_get_dimred_color_values"
XSS_LABEL = '<img src=x onerror=window.__dimredXss=1>'
XSS_DISPLAY = '<svg onload=window.__dimredXss=1>'
FORM_MODULE_SOURCE = (Path(__file__).resolve().parents[2] / "assets/js/dimred-view-form.js").read_text()


def _create_public_dimred_view(package):
    upload = FileStorage(
        stream=BytesIO(
            (
                "x,y,score,label,record\n"
                f"1,10,10,{XSS_LABEL},{XSS_DISPLAY}\n"
                "2,20,20,second,second-record\n"
                "3,30,30,third,third-record\n"
                "4,40,,fourth,fourth-record\n"
            ).encode()
        ),
        filename="colors.csv",
        content_type="text/csv",
    )
    resource = call_action(
        "resource_create",
        package_id=package["id"],
        upload=upload,
        format="csv",
        name="Colors",
    )
    view = call_action(
        "resource_view_create",
        resource_id=resource["id"],
        view_type="dimred_view",
        title="Dimred",
        method="pca",
        n_components=2,
        feature_columns=["x", "y", "score"],
        color_by="label",
        display_fields=["record", "label"],
    )
    return resource, view


def _view_url(package, resource, view):
    return tk.url_for(
        "dataset_resource.read",
        id=package["name"],
        resource_id=resource["id"],
        view_id=view["id"],
    )


def _chart_state(page):
    return page.evaluate(
        """() => {
            const container = document.getElementById("dimred-js-render");
            const chart = window.echarts && window.echarts.getInstanceByDom(container);
            if (!chart) {
                return null;
            }
            const option = chart.getOption();
            const data = option.series[0].data;
            return {
                colorValues: data.map(point => point.__colorValue),
                displayValues: data.map(point => point.__displayValues),
                itemColors: data.map(point => (point.itemStyle && point.itemStyle.color) || null),
                visualMap: option.visualMap,
                progressive: option.series[0].progressive,
                progressiveThreshold: option.series[0].progressiveThreshold,
                animationThreshold: option.series[0].animationThreshold,
            };
        }"""
    )


def _tooltip_text(page):
    return page.evaluate(
        """() => {
            const container = document.getElementById("dimred-js-render");
            const chart = window.echarts.getInstanceByDom(container);
            const option = chart.getOption();
            chart.dispatchAction({type: "showTip", seriesIndex: 0, dataIndex: 0});
            return option.tooltip[0].formatter({data: option.series[0].data[0]});
        }"""
    )


def _is_color_action(url):
    return f"/api/3/action/{COLOR_ACTION}" in url


@pytest.mark.playwright
@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_color_selector_lazy_loads_and_caches_values(page, base_url, package):
    resource, view = _create_public_dimred_view(package)
    color_requests = []
    page_errors = []
    page.add_init_script("window.__dimredXss = 0;")
    page.on("request", lambda request: color_requests.append(request.url) if _is_color_action(request.url) else None)
    page.on("pageerror", lambda error: page_errors.append(str(error)))

    page.goto(f"{base_url}{_view_url(package, resource, view)}")

    selector = page.locator("#dimred-color-select-input")
    expect(selector).to_have_value("label")
    page.wait_for_function(
        "() => window.echarts && window.echarts.getInstanceByDom(document.getElementById('dimred-js-render'))"
    )

    initial_state = _chart_state(page)
    assert initial_state["colorValues"] == [XSS_LABEL, "second", "third", "fourth"]
    assert initial_state["displayValues"] == [
        {"record": XSS_DISPLAY, "label": XSS_LABEL},
        {"record": "second-record", "label": "second"},
        {"record": "third-record", "label": "third"},
        {"record": "fourth-record", "label": "fourth"},
    ]
    assert initial_state["progressive"] == 500
    assert initial_state["progressiveThreshold"] == 5000
    assert initial_state["animationThreshold"] == 2000
    assert color_requests == []
    tooltip = _tooltip_text(page)
    assert "Source row: 1" in tooltip
    assert XSS_LABEL in tooltip
    assert XSS_DISPLAY in tooltip
    assert tooltip.count("label: " + XSS_LABEL) == 1
    assert page.evaluate("window.__dimredXss") == 0
    legend = page.locator("#dimred-color-legend")
    expect(legend).to_contain_text("Legend: label")
    expect(legend).to_contain_text(XSS_LABEL)
    summary = page.locator(".dimred-summary")
    expect(summary).to_contain_text("Sampling: All rows")
    expect(summary).to_contain_text("PCA explained variance")
    expect(page.locator(".dimred-interpretation-notes")).to_contain_text(
        "Numeric features are standardized before dimensionality reduction."
    )

    with page.expect_response(lambda response: _is_color_action(response.url) and response.ok):
        selector.select_option("score")
    page.wait_for_function("""() => {
        const chart = window.echarts.getInstanceByDom(document.getElementById("dimred-js-render"));
        return chart.getOption().visualMap.length === 1;
    }""")

    numeric_state = _chart_state(page)
    assert numeric_state["colorValues"] == [10.0, 20.0, 30.0, None]
    assert numeric_state["itemColors"] == [None, None, None, "#999999"]
    assert numeric_state["visualMap"][0]["text"] == ["score", ""]
    expect(legend).to_be_hidden()
    assert len(color_requests) == 1
    tooltip = _tooltip_text(page)
    assert tooltip.count("label: " + XSS_LABEL) == 1
    assert tooltip.count("record: " + XSS_DISPLAY) == 1

    selector.select_option("")
    selector.select_option("score")
    page.wait_for_function("""() => {
        const chart = window.echarts.getInstanceByDom(document.getElementById("dimred-js-render"));
        return chart.getOption().visualMap.length === 1;
    }""")

    assert len(color_requests) == 1
    assert page_errors == []


@pytest.mark.playwright
@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_color_selector_rejects_misaligned_values(page, base_url, package):
    resource, view = _create_public_dimred_view(package)
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))

    def corrupt_source_row_ids(route):
        response = route.fetch()
        payload = response.json()
        payload["result"]["source_row_ids"] = list(reversed(payload["result"]["source_row_ids"]))
        route.fulfill(response=response, json=payload)

    page.route(f"**/api/3/action/{COLOR_ACTION}*", corrupt_source_row_ids)
    page.goto(f"{base_url}{_view_url(package, resource, view)}")

    selector = page.locator("#dimred-color-select-input")
    expect(selector).to_have_value("label")
    with page.expect_response(lambda response: _is_color_action(response.url) and response.ok):
        selector.select_option("score")

    status = page.locator(".dimred-color-select__status")
    expect(status).to_have_text("Unable to load color values. Reload the preview.")
    assert _chart_state(page)["colorValues"] == [None, None, None, None]
    assert page_errors == []


@pytest.mark.playwright
@pytest.mark.usefixtures("clean_db", "with_plugins")
def test_method_parameter_controls_switch_reset_and_serialize(app, page, base_url):
    """Run the module against the actual CKAN-rendered resource-view form."""
    user = factories.UserWithToken()
    organization = factories.Organization(users=[{"name": user["name"], "capacity": "admin"}])
    package = factories.Dataset(owner_org=organization["id"])
    resource = call_action(
        "resource_create",
        package_id=package["id"],
        name="Columns",
        format="csv",
        upload=FileStorage(
            stream=BytesIO(b"x,y,score,label\n1,2,3,first\n"),
            filename="columns.csv",
            content_type="text/csv",
        ),
    )
    form_url = tk.url_for(
        "dataset_resource.edit_view",
        id=package["name"],
        resource_id=resource["id"],
        view_type="dimred_view",
    )
    form_response = app.get(form_url, headers={"Authorization": user["token"]})

    page.goto(base_url)
    page.set_content(
        re.sub(r"<script\b[^>]*>.*?</script>", "", form_response.text, flags=re.DOTALL | re.IGNORECASE),
        wait_until="domcontentloaded",
    )
    page.add_script_tag(content=FORM_MODULE_SOURCE)
    page.evaluate(
        """() => {
            window.ckan.module.initializeElement(document.getElementById("field-feature-columns"));
            window.ckan.module.initializeElement(document.getElementById("dimred-method-params"));
        }"""
    )

    page.locator("#field-method").select_option("tsne")
    expect(page.locator('[data-dimred-method="tsne"]')).to_be_visible()
    expect(page.locator('[data-dimred-method="pca"]')).to_be_hidden()
    expect(page.locator("#field-n-components")).to_have_value("2")

    page.locator("#field-method-param-perplexity").fill("12.5")
    expect(page.locator("#field-method-params")).to_have_value('{"random_state":42,"perplexity":12.5}')

    page.locator("#dimred-reset-method-params").click()
    expect(page.locator("#field-method-param-perplexity")).to_have_value("30")

    feature_search = page.locator("#s2id_field-feature-columns .select2-input")
    feature_search.fill("score")
    page.locator(".select2-result-label").filter(has_text="score").click()
    assert page.locator("#field-feature-columns").evaluate(
        "select => Array.from(select.selectedOptions, option => option.value)"
    ) == ["score"]

    page.locator("#field-color_by").select_option("label")
    page.locator("#dimred-select-all-features").click()
    assert page.locator("#field-feature-columns").evaluate(
        "select => Array.from(select.selectedOptions, option => option.value)"
    ) == ["x", "y", "score"]

    page.locator("#dimred-clear-features").click()
    assert page.locator("#field-feature-columns").evaluate(
        "select => Array.from(select.selectedOptions, option => option.value)"
    ) == []
