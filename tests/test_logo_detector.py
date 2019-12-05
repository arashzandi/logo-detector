import pytest

from logo_detector import __version__ # pylint: disable=maybe-no-member


def test_version():
    assert __version__ == "0.1.0"

MILKA = "https://www.ocado.com/productImages/142/14286011_0_640x640.jpg"
ADAC = "https://seeklogo.com/images/A/ADAC-logo-EA42F76A8A-seeklogo.com.png"

@pytest.mark.parametrize("image_url, label_restnet, label_cnn", [(MILKA, "milka", "fedex"), (ADAC, "ADAC", "ADAC")])
def test_detect(image_url, label_restnet, label_cnn):
    from logo_detector.detector import detect_resnet, detect_cnn
    assert detect_resnet(image_url) == label_restnet
    assert detect_cnn(image_url) == label_cnn


