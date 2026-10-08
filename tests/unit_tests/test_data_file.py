from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import numpy as np
from si.io.data_file import read_data_file, write_data_file


class TestDataFile(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.data_filename = self.folder / 'synthetic.data'
        self.data_filename.write_text('1,2,0\n3,4,1\n5,6,0\n', encoding='utf-8')

    def test_read_data_file(self):
        dataset = read_data_file(self.data_filename, sep=',', label=True)
        self.assertEqual(dataset.shape(), (3, 2))
        np.testing.assert_array_equal(dataset.X, [[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(dataset.y, [0, 1, 0])

    def test_read_data_file_no_label(self):
        dataset = read_data_file(self.data_filename, sep=',', label=False)
        self.assertEqual(dataset.shape(), (3, 3))
        self.assertFalse(dataset.has_label())
        np.testing.assert_array_equal(dataset.X, [[1, 2, 0], [3, 4, 1], [5, 6, 0]])

    def test_write_data_file(self):
        dataset = read_data_file(self.data_filename, sep=',', label=True)
        output = self.folder / 'roundtrip.data'
        write_data_file(output, dataset, sep=',', label=True)
        result = read_data_file(output, sep=',', label=True)
        np.testing.assert_array_equal(result.X, [[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result.y, [0, 1, 0])

    def test_write_data_file_no_label(self):
        dataset = read_data_file(self.data_filename, sep=',', label=False)
        output = self.folder / 'roundtrip.data'
        write_data_file(output, dataset, sep=',', label=False)
        result = read_data_file(output, sep=',', label=False)
        self.assertFalse(result.has_label())
        np.testing.assert_array_equal(result.X, [[1, 2, 0], [3, 4, 1], [5, 6, 0]])
