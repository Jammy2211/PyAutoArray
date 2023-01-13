import autoarray as aa
import autoarray.plot as aplt


def test__from_config_or_via_manual_input():

    vector_yx_quiver = aplt.VectorYXQuiver()

    assert vector_yx_quiver.config_dict["headlength"] == 0

    vector_yx_quiver = aplt.VectorYXQuiver(headlength=1)

    assert vector_yx_quiver.config_dict["headlength"] == 1

    vector_yx_quiver = aplt.VectorYXQuiver()
    vector_yx_quiver.is_for_subplot = True

    assert vector_yx_quiver.config_dict["headlength"] == 0.1

    vector_yx_quiver = aplt.VectorYXQuiver(headlength=12)
    vector_yx_quiver.is_for_subplot = True

    assert vector_yx_quiver.config_dict["headlength"] == 12


def test__quiver_vectors():

    quiver = aplt.VectorYXQuiver(
        headlength=5,
        pivot="middle",
        linewidth=3,
        units="xy",
        angles="xy",
        headwidth=6,
        alpha=1.0,
    )

    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
    )

    quiver.quiver_vectors(vectors=vectors)
