import os.path as osp
from os import makedirs
import pytest
import tempfile

import matplotlib

matplotlib.use("Agg")

from modelzilla.main import main


def test_hf_object_detection_with_plot_sink(test_assets_dir):
    cmd = [
        "-i",
        osp.join(test_assets_dir, "000000039769.jpg"),
        "-os",
        "plot",
        "HFObjectDetection",
        "--model_repo",
        "facebook/detr-resnet-50",
    ]

    main(args=cmd)


def test_hf_object_detection_with_file_sink(test_assets_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = osp.join(tmpdir, "results")
        makedirs(results_path)

        cmd = [
            "-i",
            osp.join(test_assets_dir, "000000039769.jpg"),
            "-os",
            "file",
            "-of",
            results_path,
            "HFObjectDetection",
            "--model_repo",
            "facebook/detr-resnet-50",
        ]

        main(args=cmd)
        assert osp.exists(osp.join(results_path, "000000039769.jpg"))


def test_cli_with_http_source(test_assets_dir):
    cmd = [
        "-i",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "-os",
        "plot",
        "HFObjectDetection",
        "--model_repo",
        "facebook/detr-resnet-50",
    ]

    main(args=cmd)


def test_cli_plugin_does_not_exist(test_assets_dir):
    cmd = [
        "-i",
        osp.join(test_assets_dir, "000000039769.jpg"),
        "-os",
        "plot",
        "CustomModel",
        "--model_repo",
        "facebook/detr-resnet-50",
    ]

    with pytest.raises(SystemExit):
        main(args=cmd)


def test_cli_file_sink_needs_output_path(test_assets_dir):
    cmd = [
        "-i",
        osp.join(test_assets_dir, "000000039769.jpg"),
        "-os",
        "file",
        "HFObjectDetection",
        "--model_repo",
        "facebook/detr-resnet-50",
    ]

    with pytest.raises(ValueError):
        main(args=cmd)
