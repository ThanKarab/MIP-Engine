import uuid

import pytest

from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature


@pytest.fixture(scope="session")
def localnode_id():
    return "localnode1"


def get_test_cases_get_data_model_cdes():
    test_cases_get_data_model_cdes = [
        (
            "dementia:0.1",
            186,
            CommonDataElements(
                values={
                    "dataset": CommonDataElement(
                        code="dataset",
                        label="Dataset",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "ppmi0": "PPMI_0",
                            "ppmi1": "PPMI_1",
                            "ppmi2": "PPMI_2",
                            "ppmi3": "PPMI_3",
                            "ppmi4": "PPMI_4",
                            "ppmi5": "PPMI_5",
                            "ppmi6": "PPMI_6",
                            "ppmi7": "PPMI_7",
                            "ppmi8": "PPMI_8",
                            "ppmi9": "PPMI_9",
                            "edsd0": "EDSD_0",
                            "edsd1": "EDSD_1",
                            "edsd2": "EDSD_2",
                            "edsd3": "EDSD_3",
                            "edsd4": "EDSD_4",
                            "edsd5": "EDSD_5",
                            "edsd6": "EDSD_6",
                            "edsd7": "EDSD_7",
                            "edsd8": "EDSD_8",
                            "edsd9": "EDSD_9",
                            "desd-synthdata0": "DESD-synthdata_0",
                            "desd-synthdata1": "DESD-synthdata_1",
                            "desd-synthdata2": "DESD-synthdata_2",
                            "desd-synthdata3": "DESD-synthdata_3",
                            "desd-synthdata4": "DESD-synthdata_4",
                            "desd-synthdata5": "DESD-synthdata_5",
                            "desd-synthdata6": "DESD-synthdata_6",
                            "desd-synthdata7": "DESD-synthdata_7",
                            "desd-synthdata8": "DESD-synthdata_8",
                            "desd-synthdata9": "DESD-synthdata_9",
                        },
                    ),
                    "alzheimerbroadcategory": CommonDataElement(
                        code="alzheimerbroadcategory",
                        label="Alzheimer Broad Category",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "AD": "Alzheimer’s disease",
                            "CN": "Cognitively Normal",
                            "Other": "Other",
                            "MCI": "Mild cognitive impairment",
                        },
                    ),
                    "minimentalstate": CommonDataElement(
                        code="minimentalstate",
                        label="MMSE Total scores",
                        sql_type="int",
                        is_categorical=False,
                        min=0,
                        max=30,
                    ),
                }
            ),
        ),
        (
            "tbi:0.1",
            21,
            CommonDataElements(
                values={
                    "dataset": CommonDataElement(
                        code="dataset",
                        label="Dataset",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "dummy_tbi0": "Dummy TBI_0",
                            "dummy_tbi1": "Dummy TBI_1",
                            "dummy_tbi2": "Dummy TBI_2",
                            "dummy_tbi3": "Dummy TBI_3",
                            "dummy_tbi4": "Dummy TBI_4",
                            "dummy_tbi5": "Dummy TBI_5",
                            "dummy_tbi6": "Dummy TBI_6",
                            "dummy_tbi7": "Dummy TBI_7",
                            "dummy_tbi8": "Dummy TBI_8",
                            "dummy_tbi9": "Dummy TBI_9",
                        },
                    ),
                    "gender_type": CommonDataElement(
                        code="gender_type",
                        label="Gender",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "M": "Male",
                            "F": "Female",
                        },
                    ),
                }
            ),
        ),
    ]
    return test_cases_get_data_model_cdes


@pytest.mark.parametrize(
    "data_model, expected_data_model_cdes_length, expected_data_model_cdes",
    get_test_cases_get_data_model_cdes(),
)
def test_get_data_model_cdes(
    data_model, expected_data_model_cdes_length, expected_data_model_cdes
):
    get_data_model_cdes_signature = get_celery_task_signature(
        get_celery_app("localnode1"), "get_data_model_cdes"
    )
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    data_model_cdes_json = get_data_model_cdes_signature.delay(
        request_id=request_id, data_model=data_model
    ).get()
    data_model_cdes = CommonDataElements.parse_raw(data_model_cdes_json).values
    assert len(data_model_cdes) == expected_data_model_cdes_length
    for exp_cde_code, exp_cde_metadata in expected_data_model_cdes.values.items():
        assert exp_cde_code in data_model_cdes.keys()

        assert data_model_cdes[exp_cde_code] == exp_cde_metadata
