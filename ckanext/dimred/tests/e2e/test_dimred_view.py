from __future__ import annotations

from io import BytesIO

import pytest
from playwright.sync_api import expect
from werkzeug.datastructures import FileStorage

from ckan.plugins import toolkit as tk
from ckan.tests.helpers import call_action

COLOR_ACTION = "dimred_get_dimred_color_values"
XSS_LABEL = '<img src=x onerror=window.__dimredXss=1>'


def _create_public_dimred_view(package):
    upload = FileStorage(
        stream=BytesIO(
            (
                "x,y,score,label\n"
                f"1,10,10,{XSS_LABEL}\n"
                "2,20,20,second\n"
                "3,30,30,third\n"
                "4,40,40,fourth\n"
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
                visualMap: option.visualMap,
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
    assert color_requests == []
    assert XSS_LABEL in _tooltip_text(page)
    assert page.evaluate("window.__dimredXss") == 0

    with page.expect_response(lambda response: _is_color_action(response.url) and response.ok):
        selector.select_option("score")
    page.wait_for_function("""() => {
        const chart = window.echarts.getInstanceByDom(document.getElementById("dimred-js-render"));
        return chart.getOption().visualMap.length === 1;
    }""")

    numeric_state = _chart_state(page)
    assert numeric_state["colorValues"] == [10.0, 20.0, 30.0, 40.0]
    assert len(color_requests) == 1

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
