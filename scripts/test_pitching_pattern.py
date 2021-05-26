import unittest

import pandas as pd

import pitching_pattern


class TestBallXY(unittest.TestCase):

    def test_ballXY(self):
        input_ = pd.DataFrame(
            {
                'ballX': [1, 2, 11, 3],
                'ballY': ['X', 'Y', 'D', 'D'],
            }
        )
        expected = pd.Series(['1X', '2Y', '11D', '3D'], name='ballXY')
        output = pitching_pattern.ballXY(input_)
        self.assertIsNone(pd.testing.assert_series_equal(output, expected))


class TestPitchingPattern(unittest.TestCase):

    def test_extract_pattern(self):
        # Most typical pattern
        input_ = pd.DataFrame(
            {
                'gameID': [20202173, 20202173, 20202173, 20202173],
                'inning': ['1回表', '1回表', '1回表', '1回表'],
                'batterID': [3, 3, 3, 3],
                'O': [1, 1, 1, 1],
                'pitcherID': [1, 1, 1, 1],
                'totalPitchingCount': [1, 2, 3, 4],
                'ballPositionLabel': ['内角高め', '外角低め', '内角低め', '外角低め'],
                'ballXY': ['1X', '2Y', '3D', '11D'],
                'pitchType': ['カットファストボール', '-', 'ストレート', 'ストレート'],
            }
        )
        expected = {
            'ballPositionLabel': '内角高め 外角低め 内角低め 外角低め',
            'ballXY': '1X 2Y 3D 11D',
            'pitchType': 'カットファストボール - ストレート ストレート',
        }
        output = pitching_pattern.extract_patterns(input_)
        self.assertEqual(output, expected)

    def test_extract_pattern_with_missing_count(self):

        # Change order of `totalPitchingCount` but same as typical pattern
        input_ = pd.DataFrame(
            {
                'gameID': [20202173, 20202173, 20202173, 20202173],
                'inning': ['1回表', '1回表', '1回表', '1回表'],
                'batterID': [3, 3, 3, 3],
                'O': [1, 1, 1, 1],
                'pitcherID': [1, 1, 1, 1],
                'totalPitchingCount': [1, 2, 4, 3],
                'ballPositionLabel': ['内角高め', '外角低め', '外角低め', '内角低め'],
                'ballXY': ['1X', '2Y', '11D', '3D'],
                'pitchType': ['カットファストボール', '-', 'ストレート', 'ストレート'],
            }
        )
        expected = {
            'ballPositionLabel': '内角高め 外角低め 内角低め 外角低め',
            'ballXY': '1X 2Y 3D 11D',
            'pitchType': 'カットファストボール - ストレート ストレート',
        }
        output = pitching_pattern.extract_patterns(input_)
        self.assertEqual(output, expected)

        # last `totalPitchingCount` is 5, but there aren't 3
        input_ = pd.DataFrame(
            {
                'gameID': [20202173, 20202173, 20202173, 20202173],
                'inning': ['1回表', '1回表', '1回表', '1回表'],
                'batterID': [3, 3, 3, 3],
                'O': [1, 1, 1, 1],
                'pitcherID': [1, 1, 1, 1],
                'totalPitchingCount': [1, 2, 3, 5],
                'ballPositionLabel': ['内角高め', '外角低め', '内角低め', '外角低め'],
                'ballXY': ['1X', '2Y', '3D', '11D'],
                'pitchType': ['カットファストボール', '-', 'ストレート', 'ストレート'],
            }
        )
        expected = {
            'ballPositionLabel': '内角高め 外角低め 内角低め __NO_DATA__ 外角低め',
            'ballXY': '1X 2Y 3D __NO_DATA__ 11D',
            'pitchType': 'カットファストボール - ストレート __NO_DATA__ ストレート',
        }
        output = pitching_pattern.extract_patterns(input_)
        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
