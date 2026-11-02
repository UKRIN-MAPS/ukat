import hashlib
import io
import tarfile
import zipfile
from unittest import mock

import numpy as np
import pytest
import requests

from ukat.data import fetch


class TestFetch:

    def test_asl_philips_fair_1500(self):
        # Test if the fetch function works
        image, affine = fetch.asl_philips_fair_1500()
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(image)) == 4
        assert np.shape(image) == (128, 128, 5, 50)
        assert np.shape(affine) == (4, 4)

    def test_ge_b0(self):
        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_b0(self):

        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_siemens_b0(self):
        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_siemens(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_siemens(2)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If no dataset_id is given
        with pytest.raises(TypeError):
            magnitude, phase, affine, echo_times = fetch.b0_siemens()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, echo_times = fetch.b0_siemens(3)

    def test_ge_dwi(self):
        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_philips_dwi(self):

        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_siemens_dwi(self):
        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_philips_mtr(self):
        # Test if the fetch function works
        images, affine = fetch.mtr_philips()

        # Check the format of the outputs
        assert isinstance(images, np.ndarray)
        assert np.unique(np.isnan(images)) != [True]
        assert isinstance(affine, np.ndarray)
        assert np.shape(images)[-1] == 2
        assert len(np.shape(images)) == 3
        assert np.shape(affine) == (4, 4)

    def test_philips_pc_left(self):
        # Test if the fetch function works
        magnitude, phase, mask, affine, \
            velocity_encoding = fetch.phase_contrast_left_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(mask, np.ndarray)
        assert len(np.unique(mask)) == 2
        assert isinstance(affine, np.ndarray)
        assert isinstance(velocity_encoding, int)
        assert len(np.shape(magnitude)) == 3
        assert len(np.shape(phase)) == 3
        assert len(np.shape(mask)) == 3
        assert np.shape(affine) == (4, 4)

    def test_philips_pc_right(self):
        # Test if the fetch function works
        magnitude, phase, mask, affine, \
            velocity_encoding = fetch.phase_contrast_right_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(mask, np.ndarray)
        assert len(np.unique(mask)) == 2
        assert isinstance(affine, np.ndarray)
        assert isinstance(velocity_encoding, int)
        assert len(np.shape(magnitude)) == 3
        assert len(np.shape(phase)) == 3
        assert len(np.shape(mask)) == 3
        assert np.shape(affine) == (4, 4)

    def test_philips_t1(self):
        # Test if the fetch function works
        magnitude, phase, affine, inversion_times, tss = fetch.t1_philips(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(inversion_times, np.ndarray)
        assert isinstance(tss, int)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(inversion_times)) == 1

        # Test if the fetch function works
        magnitude, phase, affine, inversion_times, tss = fetch.t1_philips(2)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(inversion_times, np.ndarray)
        assert isinstance(tss, float)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(inversion_times)) == 1

        # If no dataset_id is given
        with pytest.raises(TypeError):
            magnitude, phase, affine, inversion_times, _ = \
                fetch.t1_philips()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, inversion_times, _ = \
                fetch.t1_philips(3)

    def test_philips_t1_molli(self):
        # Test if the fetch function works
        image, affine, inversion_times = fetch.t1_molli_philips()

        # Check the format of the outputs
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert len(np.shape(image)) == 4
        assert isinstance(affine, np.ndarray)
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(inversion_times)) == 1

    def test_philips_t1w(self):
        # Test if the fetch function works
        image, affine = fetch.t1w_volume_philips()

        # Check the format of the outputs
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(image)) == 3
        assert np.shape(affine) == (4, 4)

    def test_ge_t2(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, affine, echo_times = fetch.t2_ge(2)

    def test_philips_t2(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2_philips(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2_philips(2)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, affine, echo_times = fetch.t2_philips(3)

    def test_siemens_t2(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, affine, echo_times = fetch.t2_siemens(2)

    def test_ge_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_siemens_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_t2w(self):
        # Test if the fetch function works
        image, affine = fetch.t2w_volume_philips()

        # Check the format of the outputs
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(image)) == 3
        assert np.shape(affine) == (4, 4)

    def test_philips_tsnr(self):
        # Test if the fetch function works with high tSNR data
        data, affine = fetch.tsnr_high_philips()

        # Check the format of the outputs
        assert isinstance(data, np.ndarray)
        assert np.unique(np.isnan(data)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(data)) == 4
        assert np.shape(affine) == (4, 4)

        # Test if the fetch function works with low tSNR data
        data, affine = fetch.tsnr_low_philips()

        # Check the format of the outputs
        assert isinstance(data, np.ndarray)
        assert np.unique(np.isnan(data)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(data)) == 4
        assert np.shape(affine) == (4, 4)


def _fake_response(status=200, content=b'', headers=None):
    response = mock.MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.iter_content.return_value = [content]
    if status >= 400 and status not in fetch._RETRY_STATUS_CODES:
        response.raise_for_status.side_effect = requests.HTTPError(
            f'{status} error')
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestZenodoFetcher:
    """Tests of the Zenodo API downloader that do not use the network."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(fetch.time, 'sleep', lambda s: None)

    def test_record_id_from_url(self):
        for url in ['https://zenodo.org/record/4757887/files/',
                    'https://zenodo.org/records/4757887/files/',
                    'https://zenodo.org/record/4757887']:
            assert fetch._record_id_from_url(url) == '4757887'
        with pytest.raises(ValueError):
            fetch._record_id_from_url('https://example.com/files/')

    def test_token_from_environment(self, monkeypatch):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        assert fetch._get_zenodo_token() is None
        monkeypatch.setenv('ZENODO_TOKEN', '   ')
        assert fetch._get_zenodo_token() is None
        monkeypatch.setenv('ZENODO_TOKEN', ' abc123 ')
        assert fetch._get_zenodo_token() == 'abc123'

    def test_download_uses_api_url_without_token(self, monkeypatch,
                                                tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        target = str(tmp_path / 'file.bin')
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=b'data')) as get:
            fetch._download_file('123', 'file.bin', target)
        args, kwargs = get.call_args
        assert args[0] == 'https://zenodo.org/api/records/123/files/' \
                          'file.bin/content'
        assert 'Authorization' not in kwargs['headers']
        with open(target, 'rb') as f:
            assert f.read() == b'data'
        assert not (tmp_path / 'file.bin.part').exists()

    def test_download_sends_token_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv('ZENODO_TOKEN', 'secret')
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=b'x')) as get:
            fetch._download_file('123', 'f', str(tmp_path / 'f'))
        assert get.call_args[1]['headers'] == {
            'Authorization': 'Bearer secret'}

    def test_download_recovers_from_interrupted_stream(self, monkeypatch,
                                                       tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        target = tmp_path / 'f'

        def broken_stream(chunk_size):
            yield b'partial'
            raise requests.exceptions.ChunkedEncodingError('interrupted')

        interrupted = _fake_response()
        interrupted.iter_content.side_effect = broken_stream
        responses = [interrupted, _fake_response(content=b'complete')]
        with mock.patch.object(fetch.requests, 'get',
                               side_effect=responses) as get:
            fetch._download_file('1', 'f', str(target))

        assert get.call_count == 2
        # Only the complete download is kept and no partial files remain
        assert target.read_bytes() == b'complete'
        assert list(tmp_path.iterdir()) == [target]

    def test_download_interrupted_stream_leaves_no_partial_file(
            self, monkeypatch, tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)

        def broken_stream(chunk_size):
            yield b'partial'
            raise requests.exceptions.ChunkedEncodingError('interrupted')

        def make_broken(*args, **kwargs):
            response = _fake_response()
            response.iter_content.side_effect = broken_stream
            return response

        with mock.patch.object(fetch.requests, 'get',
                               side_effect=make_broken):
            with pytest.raises(fetch.FetcherError):
                fetch._download_file('1', 'f', str(tmp_path / 'f'))
        assert list(tmp_path.iterdir()) == []

    def test_download_retries_then_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        responses = [_fake_response(503),
                     _fake_response(429, headers={'Retry-After': '1'}),
                     requests.ConnectionError(),
                     _fake_response(content=b'ok')]
        with mock.patch.object(fetch.requests, 'get',
                               side_effect=responses) as get:
            fetch._download_file('1', 'f', str(tmp_path / 'f'))
        assert get.call_count == 4
        assert (tmp_path / 'f').read_bytes() == b'ok'

    def test_download_gives_up_after_max_attempts(self, monkeypatch,
                                                  tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        with mock.patch.object(fetch.requests, 'get',
                               side_effect=lambda *a, **k: _fake_response(
                                   503)) as get:
            with pytest.raises(fetch.FetcherError):
                fetch._download_file('1', 'f', str(tmp_path / 'f'))
        assert get.call_count == fetch._MAX_ATTEMPTS
        assert not (tmp_path / 'f').exists()

    def test_download_does_not_retry_client_errors(self, monkeypatch,
                                                   tmp_path):
        monkeypatch.delenv('ZENODO_TOKEN', raising=False)
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(404)) as get:
            with pytest.raises(fetch.FetcherError):
                fetch._download_file('1', 'f', str(tmp_path / 'f'))
        assert get.call_count == 1

    def test_fetcher_downloads_and_skips_cached_files(self, tmp_path):
        content = b'hello'
        md5 = hashlib.md5(content).hexdigest()
        fetcher = fetch._make_fetcher(
            'test', str(tmp_path / 'data'),
            'https://zenodo.org/record/42/files/', ['a.txt'], ['b.txt'],
            md5_list=[md5])
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=content)) as get:
            files, folder = fetcher()
            assert get.call_count == 1
            assert get.call_args[0][0].endswith('/42/files/a.txt/content')
            assert files['b.txt'][1] == md5
            assert (tmp_path / 'data' / 'b.txt').read_bytes() == content
            # Second call finds the valid cached file and does not download
            fetcher()
            assert get.call_count == 1

    def test_fetcher_redownloads_on_bad_cached_file(self, tmp_path):
        content = b'good'
        folder = tmp_path / 'data'
        folder.mkdir()
        (folder / 'a.txt').write_bytes(b'corrupt')
        fetcher = fetch._make_fetcher(
            'test', str(folder), 'https://zenodo.org/record/42/files/',
            ['a.txt'], ['a.txt'],
            md5_list=[hashlib.md5(content).hexdigest()])
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=content)) as get:
            fetcher()
        assert get.call_count == 1
        assert (folder / 'a.txt').read_bytes() == content

    def test_fetcher_checksum_mismatch_raises(self, tmp_path):
        fetcher = fetch._make_fetcher(
            'test', str(tmp_path), 'https://zenodo.org/record/42/files/',
            ['a.txt'], ['a.txt'], md5_list=['0' * 32])
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=b'something')):
            with pytest.raises(fetch.FetcherError):
                fetcher()

    def test_fetcher_unzips(self, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('inner.txt', 'hi')
        content = buffer.getvalue()
        fetcher = fetch._make_fetcher(
            'test', str(tmp_path / 'data'),
            'https://zenodo.org/record/42/files/', ['a.zip'], ['a.zip'],
            md5_list=[hashlib.md5(content).hexdigest()], unzip=True)
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=content)):
            files, folder = fetcher()
        assert files['a.zip'][2] == ('inner.txt',)
        assert (tmp_path / 'data' / 'inner.txt').read_text() == 'hi'

    def test_fetcher_untars(self, tmp_path):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as t:
            info = tarfile.TarInfo('inner.txt')
            info.size = 2
            t.addfile(info, io.BytesIO(b'hi'))
        fetcher = fetch._make_fetcher(
            'test', str(tmp_path / 'data'),
            'https://zenodo.org/record/42/files/', ['a.tar.gz'],
            ['a.tar.gz'], unzip=True)
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(
                                   content=buffer.getvalue())):
            fetcher()
        assert (tmp_path / 'data' / 'inner.txt').read_text() == 'hi'

    def test_fetcher_unknown_archive_extension_raises(self, tmp_path):
        fetcher = fetch._make_fetcher(
            'test', str(tmp_path), 'https://zenodo.org/record/42/files/',
            ['a.rar'], ['a.rar'], unzip=True)
        with mock.patch.object(fetch.requests, 'get',
                               return_value=_fake_response(content=b'x')):
            with pytest.raises(ValueError):
                fetcher()
