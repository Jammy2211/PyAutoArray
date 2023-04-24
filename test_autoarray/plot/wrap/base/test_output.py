import autoarray.plot as aplt

from os import path

import shutil

directory = path.dirname(path.realpath(__file__))


def test__constructor():
    output = aplt.Output()

    assert output.path == None
    assert output._format == None
    assert output.format == "show"
    assert output.filename == None

    output = aplt.Output(path="Path", format="png", filename="file")

    assert output.path == "Path"
    assert output._format == "png"
    assert output.format == "png"
    assert output.filename == "file"

    if path.exists(output.path):
        shutil.rmtree(output.path)


def test__input_path_is_created():
    test_path = path.join(directory, "files", "output_path")

    if path.exists(test_path):
        shutil.rmtree(test_path)

    assert not path.exists(test_path)

    output = aplt.Output(path=test_path)

    assert path.exists(test_path)
