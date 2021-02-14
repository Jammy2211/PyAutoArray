import autoarray as aa


class TestExposureInfo:
    def test__exposure_info_has_date_and_time_of_observation__calcs_julian_date(self):

        exposure_info = aa.ExposureInfo(
            date_of_observation="2000-01-01", time_of_observation="00:00:00"
        )

        assert exposure_info.modified_julian_date == 51544.0
