import autoarray as aa


class TestHeader:
    def test__header_has_date_and_time_of_observation__calcs_julian_date(self):

        header_sci_obj = {"DATE-OBS": "2000-01-01", "TIME-OBS": "00:00:00"}

        header = aa.Header(header_sci_obj=header_sci_obj, header_hdu_obj=None)

        assert header.modified_julian_date == 51544.0
