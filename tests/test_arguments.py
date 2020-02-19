import sys
import mock
import pytest
from ml4cvd.arguments import parse_args, TMAPS
from ml4cvd.test_utils import TMAPS as MOCK_TMAPS


@mock.patch.dict(TMAPS, MOCK_TMAPS)
class TestUConnect:

    def test_no_u(self, tmpdir):
        sys.argv = [
            'train',
            '--output_folder', str(tmpdir),
        ]
        args = parse_args()
        assert len(args.u_connect) == 0

    def test_simple_u(self, tmpdir):
        inp_key = '3d_cont'
        sys.argv = [
            'train',
            '--output_folder', str(tmpdir),
            '--input_tensors', inp_key,
            '--output_tensors', inp_key,
            '--u_connect', inp_key, inp_key,
        ]
        args = parse_args()
        assert len(args.u_connect) == 1
        inp, out = list(args.u_connect.items())[0]
        tmap = MOCK_TMAPS[inp_key]
        assert inp == tmap
        assert out == {tmap, }

    def test_bad_u(self, tmpdir):
        inp_key1 = '3d_cont'
        inp_key2 = '3d_cat'
        sys.argv = [
            'train',
            '--output_folder', str(tmpdir),
            '--input_tensors', inp_key1, inp_key2,
            '--output_tensors', inp_key1,
            '--u_connect', inp_key1, inp_key1,
            '--u_connect', inp_key2, inp_key1,
        ]
        with pytest.raises(ValueError):
            parse_args()
