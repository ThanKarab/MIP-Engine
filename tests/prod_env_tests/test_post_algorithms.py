import json
import re

import numpy as np
import pytest
import requests

from tests.prod_env_tests import algorithms_url


def get_parametrization_list_success_cases():
    parametrization_list = []
    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "x": [
                "lefthippocampus",
                "righthippocampus",
                "rightppplanumpolare",
                "leftamygdala",
                "rightamygdala",
            ],
            "y": ["alzheimerbroadcategory"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                                "righthippocampus",
                                "rightppplanumpolare",
                                "leftamygdala",
                                "rightamygdala",
                                "alzheimerbroadcategory",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "parameters": {"classes": ["AD", "CN"]},
    }
    expected_response = {
        "title": "Logistic Regression Coefficients",
        "columns": [
            {
                "name": "variable",
                "type": "STR",
                "data": [
                    "lefthippocampus",
                    "righthippocampus",
                    "rightppplanumpolare",
                    "leftamygdala",
                    "rightamygdala",
                ],
            },
            {
                "name": "coefficient",
                "type": "FLOAT",
                "data": [
                    -2.6061540990015115,
                    3.6188386684744067,
                    3.4370357819153647,
                    -3.374943651722157,
                    -11.190272547103902,
                ],
            },
        ],
    }
    parametrization_list.append((algorithm_name, request_dict, expected_response))
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_success_cases(),
)
def test_post_algorithms(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"

    response = response.json()
    columns = response["columns"]
    expected_columns = expected_response["columns"]

    for column, expected_column in zip(columns, expected_columns):
        assert column["name"] == expected_column["name"]
        assert column["type"] == expected_column["type"]
        if column["type"] == "STR":
            assert column["data"] == expected_column["data"]
        elif column["type"] == "FLOAT":
            np.testing.assert_allclose(column["data"], expected_column["data"])


def get_parametrization_list_exception_cases():
    parametrization_list = []

    # ~~~~~~~~~~exception case 1~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "wrong_name": {
            "data_model": "dementia:0.1",
            "datasets": ["test_dataset1", "test_dataset2"],
            "x": ["test_cde1", "test_cde2"],
            "y": ["test_cde3"],
        }
    }
    expected_response = (400, ".*Algorithm execution request malformed")
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    # ~~~~~~~~~~exception case 2~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "inputdata": {
            "data_model": "non_existing",
            "datasets": ["test_dataset1", "test_dataset2"],
            "x": ["test_cde1", "test_cde2"],
            "y": ["test_cde3"],
        },
    }

    expected_response = (460, "Data model .* does not exist.*")
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    # ~~~~~~~~~~exception case 3~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "x": ["lefthippocampus"],
            "y": ["alzheimerbroadcategory"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "condition": "OR",
                        "rules": [
                            {
                                "id": "subjectage",
                                "field": "subjectage",
                                "type": "real",
                                "input": "number",
                                "operator": "greater",
                                "value": 200.0,
                            }
                        ],
                    }
                ],
                "valid": True,
            },
        },
        "parameters": {"classes": ["AD", "CN"]},
    }

    expected_response = (
        461,
        "The algorithm could not run with the input provided because there are insufficient data.",
    )
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_exception_cases(),
)
def test_post_algorithm_error(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    request_json = json.dumps(request_dict)
    response = requests.post(algorithm_url, data=request_json, headers=headers)
    exp_response_status, exp_response_message = expected_response
    assert (
        response.status_code == exp_response_status
    ), f"Response message: {response.text}"
    assert re.search(exp_response_message, response.text)
