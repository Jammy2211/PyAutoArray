import autoarray as aa
import autoarray.plot as aplt

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
